"""Per-iteration step-through diagnostics for the C=1 Gibbs sampler.

Used by `notebooks/jack/gibbs-diagnostic.ipynb`. Under the two-track design
(specs/model.md §4) the C=0 hypothesis needs no sampling — it is one exact
Kalman smoother call — so there is nothing to step through there. This module
steps through the **C=1** sampler, whose three blocks per sweep are:

    Block A.  z^EV | Theta, z^LDS   — HMM forward-filter backward-sample.
    Block B.  Theta_k | z^EV, z^LDS — truncated-Normal conjugate (k ∈ {low,high}).
    Block C.  z^LDS | z^EV, Theta   — Kalman FFBS on residuals x - Theta[z^EV].

`infer_home_c1` runs these and returns only the final state; this module
re-implements one sweep as a *stepper* exposing the intermediate quantities
(the HMM state marginals, the Theta sufficient statistics, the z^LDS update)
so each block can be inspected by eye.

z^EV heatmaps use one viridis colour per state (off=purple, low=teal,
high=yellow). Alongside each discrete draw we show the "fuzzy posterior": the
per-cell HMM marginal γ_{d,t}(k), rendered as the mass-weighted blend of the
three state colours (e.g. 0.1·purple + 0.2·teal + 0.7·yellow).

Layout:
    Section 1   IterRecord + step_one_iter   — the stepper
    Section 2   run_silent                   — fast inner loop, no recording
    Section 3   z^EV colour helpers          — discrete + fuzzy-blend
    Section 4   plotters
    Section 5   narrate_iteration            — prints + plots one iteration
    Section 6   plot_traces + run_diagnostic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from models import graphical_model as gm
from models.graphical_model import ev, non_ev_lds
from models.graphical_model.params import (
    K, STATE_NAMES, T, THETA_BOUNDS, THETA_VAR_FLOOR, ModelParams,
)
from models.graphical_model.inference import compute_loglik


# ===========================================================================
# Section 1.  IterRecord + step_one_iter
# ===========================================================================

@dataclass
class IterRecord:
    """All intermediate quantities produced by one C=1 Gibbs sweep."""
    iter_idx: int

    # Block A: z^EV | Theta, z^LDS  ────────────────────────────────────────
    z_ev_before:     np.ndarray             # (D, T)  state at start of sweep
    z_ev_after:      np.ndarray             # (D, T)  newly drawn sample
    z_ev_marginals:  np.ndarray             # (D, T, K)  γ_{d,t}(k) = p(z=k | x, Θ, z^LDS)

    # Block B: Theta_k for k ∈ {low, high}  ────────────────────────────────
    theta_before:    np.ndarray             # (K,)
    theta_after:     np.ndarray             # (K,)
    theta_posterior: dict                   # {k: {'m','sd','lb','ub','n_cells','S_r','S_inv_var',...}}

    # Block C: z^LDS  ──────────────────────────────────────────────────────
    nonev_mean_before: np.ndarray           # (D, T)  C_lds @ z_lds_prev
    nonev_mean_after:  np.ndarray           # (D, T)  C_lds @ z_lds_new

    # End-of-sweep scalars  ─────────────────────────────────────────────────
    logL_after:      float                  # complete-data emission logL at (z_ev, Θ, z_lds_prev)


@dataclass
class ChainState:
    """Mutable state of the C=1 chain between sweeps (no C — it is fixed to 1)."""
    z_lds:   np.ndarray            # (D, L)
    z_ev:    np.ndarray            # (D, T)  int
    theta:   np.ndarray            # (K,)


def init_state(home_x: np.ndarray, params: ModelParams) -> ChainState:
    """Cold-start: z_lds = smoother mean on home_x (treats x as Non-EV);
    z_ev = all-off; theta = prior mean."""
    D, _ = home_x.shape
    z_lds = params.lds.smooth(home_x).z_smooth                  # (D, L)
    return ChainState(
        z_lds = z_lds,
        z_ev  = np.zeros((D, T), dtype=np.int64),
        theta = params.mu_theta.copy(),
    )


def step_one_iter(
    state:    ChainState,
    home_x:   np.ndarray,
    params:   ModelParams,
    log_pi:   np.ndarray,
    log_P:    np.ndarray,
    rng:      np.random.Generator,
    iter_idx: int,
) -> IterRecord:
    """Run one C=1 Gibbs sweep on `state` IN PLACE and return a diagnostic record.

    Mirrors `infer_home_c1`'s main loop but additionally computes the smoothed
    HMM marginals (the fuzzy posterior) before drawing z^EV.
    """
    lds       = params.lds
    C_lds     = lds.C
    nonev_var = lds.diag_R()                # (T,)

    z_ev_before        = state.z_ev.copy()
    theta_before       = state.theta.copy()
    nonev_mean_before  = state.z_lds @ C_lds.T                  # (D, T)

    # ── Block A: z^EV | Theta, z^LDS ────────────────────────────────────────
    # Fuzzy posterior (smoothed marginals) + one FFBS draw from the same model.
    z_ev_marginals = ev.hmm_marginals(home_x, state.theta, nonev_mean_before,
                                       nonev_var, params, log_pi, log_P)
    log_f, _   = ev.hmm_forward(home_x, state.theta, nonev_mean_before,
                                nonev_var, params, log_pi, log_P)
    z_ev_after = ev.hmm_backward_sample(log_f, params, rng)

    # ── Block B: Theta_k for k ∈ {low, high} ────────────────────────────────
    theta_after     = state.theta.copy()
    theta_posterior: dict = {}
    for k in (1, 2):
        post = _theta_k_posterior(home_x, z_ev_after, nonev_mean_before,
                                   nonev_var, params, k)
        theta_after[k] = ev._truncnorm_sample(post['m'], post['sd'],
                                               post['lb'], post['ub'], rng)
        theta_posterior[k] = post

    # ── Block C: z^LDS | z^EV, Theta ────────────────────────────────────────
    z_lds_after      = non_ev_lds.sample_z_lds(home_x, z_ev_after, theta_after,
                                                params.sigma2_ev, lds, rng)
    nonev_mean_after = z_lds_after @ C_lds.T                    # (D, T)

    logL_after = compute_loglik(home_x, z_ev_after, theta_after,
                                 nonev_mean_before, nonev_var, params)

    # Commit
    state.z_lds  = z_lds_after
    state.z_ev   = z_ev_after
    state.theta  = theta_after

    return IterRecord(
        iter_idx          = iter_idx,
        z_ev_before       = z_ev_before,
        z_ev_after        = z_ev_after,
        z_ev_marginals    = z_ev_marginals,
        theta_before      = theta_before,
        theta_after       = theta_after,
        theta_posterior   = theta_posterior,
        nonev_mean_before = nonev_mean_before,
        nonev_mean_after  = nonev_mean_after,
        logL_after        = float(logL_after),
    )


def _theta_k_posterior(
    x:          np.ndarray,
    z:          np.ndarray,
    nonev_mean: np.ndarray,
    nonev_var:  np.ndarray,
    params:     ModelParams,
    k:          int,
) -> dict:
    """Closed-form truncated-Normal posterior parameters for Theta_k.

    Mirrors the math inside `ev.sample_theta_k` but returns (m, sd, lb, ub) and
    the sufficient statistics — without drawing.
    """
    sigma2_ev_k = params.sigma2_ev[k]
    sig2_prior  = max(params.sigma2_theta[k], THETA_VAR_FLOOR)
    lb, ub      = THETA_BOUNDS[k]

    mask = (z == k)
    n_cells = int(mask.sum())
    if n_cells == 0:
        return dict(m=float(params.mu_theta[k]), sd=float(np.sqrt(sig2_prior)),
                    lb=lb, ub=ub, n_cells=0, S_r=0.0, S_inv_var=0.0,
                    used_prior_only=True)

    var_t     = sigma2_ev_k + nonev_var                  # (T,)
    inv_var_t = 1.0 / var_t
    r         = x - nonev_mean
    S_inv_var = float((mask * inv_var_t[None, :]).sum())
    S_r       = float((mask * r * inv_var_t[None, :]).sum())

    prec = 1.0 / sig2_prior + S_inv_var
    m    = (params.mu_theta[k] / sig2_prior + S_r) / prec
    return dict(m=float(m), sd=float(np.sqrt(1.0 / prec)),
                lb=lb, ub=ub, n_cells=n_cells,
                S_r=S_r, S_inv_var=S_inv_var,
                used_prior_only=False)


# ===========================================================================
# Section 2.  Silent inner loop
# ===========================================================================

def run_silent(
    state:    ChainState,
    home_x:   np.ndarray,
    params:   ModelParams,
    log_pi:   np.ndarray,
    log_P:    np.ndarray,
    rng:      np.random.Generator,
    n_iters:  int,
) -> None:
    """Advance the C=1 chain by `n_iters` sweeps without recording — used to
    fast-forward through burn-in before the narrated range."""
    lds       = params.lds
    C_lds     = lds.C
    nonev_var = lds.diag_R()

    for _ in range(n_iters):
        nonev_mean = state.z_lds @ C_lds.T
        state.z_ev, _ = ev.ffbs(home_x, state.theta, nonev_mean, nonev_var,
                                params, log_pi, log_P, rng)
        for k in (1, 2):
            state.theta[k] = ev.sample_theta_k(home_x, state.z_ev, nonev_mean,
                                                nonev_var, params, k, rng)
        state.z_lds = non_ev_lds.sample_z_lds(home_x, state.z_ev, state.theta,
                                               params.sigma2_ev, lds, rng)


# ===========================================================================
# Section 3.  z^EV colour helpers — one colour per state + fuzzy blend
# ===========================================================================
# Viridis-inspired: off=deep purple, low=teal, high=yellow. No correctness
# tinting — a cell's colour is purely its state. The "fuzzy posterior" panels
# render the per-cell HMM marginal as the mass-weighted blend of these colours.

Z_EV_STATE_COLORS = ["#440154", "#21918c", "#fde725"]   # off, low, high
_BASE_RGB = np.array([to_rgb(c) for c in Z_EV_STATE_COLORS])   # (K, 3) in [0,1]


def z_ev_discrete_rgb(z: np.ndarray) -> np.ndarray:
    """Map integer states (...,) → RGB (..., 3) using one colour per state."""
    return _BASE_RGB[z]


def z_ev_blend_rgb(marginals: np.ndarray) -> np.ndarray:
    """Map a categorical posterior (..., K) → blended RGB (..., 3):

        colour = Σ_k marginals[..., k] · base_colour[k]
    """
    return np.clip(marginals @ _BASE_RGB, 0.0, 1.0)


def add_z_ev_legend(fig, y: float = -0.04) -> None:
    """Add the 3-state colour legend once per figure."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=name)
               for c, name in zip(Z_EV_STATE_COLORS, STATE_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, y), fontsize=9, frameon=False,
               title="EV state colour (fuzzy panels blend these by posterior mass)")


# ===========================================================================
# Section 4.  Plotters
# ===========================================================================

def _imshow_rgb_strip(ax, rgb_row: np.ndarray, title: str | None = None) -> None:
    """Render a (T, 3) RGB row as a 1-row image spanning 0..24 h."""
    ax.imshow(rgb_row[None, :, :], aspect="auto",
              extent=[0, 24, 0, 1], interpolation="nearest")
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.set_yticks([])


def plot_z_ev_per_day_grid(
    fig,
    z_true:        np.ndarray,    # (D, T)      int
    z_pred:        np.ndarray,    # (D, T)      int   (the drawn sample)
    z_marginals:   np.ndarray,    # (D, T, K)   fuzzy posterior
    days_to_show:  Sequence[int],
    iter_idx:      int,
) -> None:
    """`(len(days), 3)` grid of per-day strips: ground truth | sample | posterior.

    - ground truth & sample are discrete (one colour per state)
    - posterior is the mass-weighted blend of the state colours
    """
    n = len(days_to_show)
    if n == 0:
        return
    axes = fig.subplots(n, 3, sharex=True, sharey=False, squeeze=False)
    col_titles = ["ground truth", f"sample (iter {iter_idx})", "posterior γ (fuzzy)"]
    for i, d in enumerate(days_to_show):
        _imshow_rgb_strip(axes[i, 0], z_ev_discrete_rgb(z_true[d]))
        _imshow_rgb_strip(axes[i, 1], z_ev_discrete_rgb(z_pred[d]))
        _imshow_rgb_strip(axes[i, 2], z_ev_blend_rgb(z_marginals[d]))
        axes[i, 0].set_ylabel(f"day {d}", fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=10)
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=10)
        if i == n - 1:
            for j in range(3):
                axes[i, j].set_xlabel("hour of day")
    add_z_ev_legend(fig, y=-0.04)


def plot_z_ev_full(
    fig,
    z_true:      np.ndarray,    # (D, T)     int
    z_pred:      np.ndarray,    # (D, T)     int
    z_marginals: np.ndarray,    # (D, T, K)  fuzzy posterior
    title:       str,
) -> None:
    """Three full-D (day × hour) heatmaps: truth | MAP/sample | posterior blend."""
    axes = fig.subplots(1, 3, sharex=True, sharey=True, squeeze=True)
    D = z_true.shape[0]
    for ax, img, t in [
        (axes[0], z_ev_discrete_rgb(z_true), "ground truth"),
        (axes[1], z_ev_discrete_rgb(z_pred), "MAP / sample"),
        (axes[2], z_ev_blend_rgb(z_marginals), "posterior γ (fuzzy)"),
    ]:
        ax.imshow(img, aspect="auto", extent=[0, 24, D, 0], interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("hour of day")
    axes[0].set_ylabel("day")
    fig.suptitle(title, fontsize=11)
    add_z_ev_legend(fig, y=-0.02)


def plot_power_decomposition(
    axes,                              # (3,) array of axes
    home_x:        np.ndarray,         # (D, T)   total observed grid power
    x_nev_true:    np.ndarray,         # (D, T)   ground-truth Non-EV
    x_ev_true:     np.ndarray,         # (D, T)   ground-truth EV
    nonev_pred:    np.ndarray,         # (D, T)   (C z^LDS)[d, t]
    z_ev:          np.ndarray,         # (D, T)   sampled EV state per cell
    theta:         np.ndarray,         # (K,)     current EV magnitudes
    days_to_show:  Sequence[int],
    iter_idx:      int,
) -> None:
    """Three stacked plots over `days_to_show`: Non-EV / EV / Total, true vs pred."""
    if len(days_to_show) == 0:
        return

    ev_pred = theta[z_ev]                                      # (D, T)
    seg_x   = np.arange(T) * 15 / 60                            # 0..24
    xs      = np.concatenate([seg_x + 24 * i for i in range(len(days_to_show))])
    offsets = [24 * i for i in range(len(days_to_show))]

    def _concat(arr2d):
        return np.concatenate([arr2d[d] for d in days_to_show])

    panels = [
        (axes[0], "Non-EV power", _concat(x_nev_true), _concat(nonev_pred)),
        (axes[1], "EV power",     _concat(x_ev_true),  _concat(ev_pred)),
        (axes[2], "Total power",  _concat(home_x),     _concat(nonev_pred + ev_pred)),
    ]
    for ax, title, truth, pred in panels:
        ax.plot(xs, truth, lw=1.2, color="C0", label="true")
        ax.plot(xs, pred,  lw=1.4, color="C3", label="predicted")
        for off in offsets[1:]:
            ax.axvline(off, color="black", lw=1.8, alpha=0.9)
        for d, off in zip(days_to_show, offsets):
            ax.text(off + 12, 1.02, f"day {d}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10,
                    color="red", fontweight="bold")
        ax.set_xticks(offsets)
        ax.set_xticklabels([f"d{d}" for d in days_to_show], fontsize=8)
        ax.set_ylabel("kW")
        ax.set_title(f"{title}  —  iter {iter_idx}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)


def plot_z_lds_compare(
    ax,
    home_x:        np.ndarray,
    nonev_before:  np.ndarray,
    nonev_after:   np.ndarray,
    days_to_show:  Sequence[int],
    title:         str,
) -> None:
    """Concatenate `days_to_show` along x; show x, nonev_before, nonev_after."""
    if len(days_to_show) == 0:
        ax.set_title(title + "  (no days selected)")
        return

    segments_x, segments_x_data, segments_b, segments_a = [], [], [], []
    offsets = [0]
    for d in days_to_show:
        seg_x = np.arange(T) * 15 / 60
        segments_x.append(seg_x + offsets[-1])
        segments_x_data.append(home_x[d])
        segments_b.append(nonev_before[d])
        segments_a.append(nonev_after[d])
        offsets.append(offsets[-1] + 24)

    xs       = np.concatenate(segments_x)
    x_data   = np.concatenate(segments_x_data)
    nonev_b  = np.concatenate(segments_b)
    nonev_a  = np.concatenate(segments_a)

    ax.plot(xs, x_data,  lw=1.0, color="C0",            label=r"$x$ (observed)")
    ax.plot(xs, nonev_b, lw=1.4, color="C7", alpha=0.8, label=r"$\hat z^{LDS}$ before")
    ax.plot(xs, nonev_a, lw=1.4, color="C1",            label=r"$\hat z^{LDS}$ after")

    for d, off in zip(days_to_show, offsets[:-1]):
        if off > 0:
            ax.axvline(off, color="black", lw=1.8, alpha=0.9)
        ax.text(off + 12, 1.02, f"day {d}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10,
                color="red", fontweight="bold")
    ax.set_xticks(offsets[:-1])
    ax.set_xticklabels([f"d{d}" for d in days_to_show], fontsize=8)
    ax.set_ylabel("kW")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)


def plot_theta_posterior(ax, record: IterRecord, k: int) -> None:
    """For state k: prior mean, posterior mean ± sd, truncation bounds, drawn value."""
    post   = record.theta_posterior[k]
    m, sd  = post["m"], post["sd"]
    lb, ub = post["lb"], post["ub"]
    drawn  = record.theta_after[k]
    before = record.theta_before[k]

    ax.set_title(f"Θ_{STATE_NAMES[k]}  posterior  (n_cells={post['n_cells']})", fontsize=10)
    ub_plot = min(ub, m + 4 * sd + 1) if np.isfinite(ub) else (m + 4 * sd + 1)
    ax.axvspan(lb, ub_plot, color="#eef", alpha=0.5,
               label=f"truncation [{lb}, {'∞' if not np.isfinite(ub) else ub}]")
    xs = np.linspace(max(lb - 0.5, m - 4*sd), min(ub_plot, m + 4*sd), 200)
    pdf = np.exp(-0.5 * ((xs - m) / sd) ** 2)
    pdf = pdf / pdf.max() * 0.8
    ax.plot(xs, pdf, color="C0", lw=1.5, label="posterior (truncated)")
    ax.axvline(m,      color="C0", lw=1.2, label=f"post. mean = {m:.3f}")
    ax.axvline(drawn,  color="C3", lw=1.8, label=f"drawn = {drawn:.3f}")
    ax.axvline(before, color="C7", lw=1.0, ls="--", label=f"before = {before:.3f}")
    ax.set_xlabel("Θ (kW)")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)


def plot_traces(ax_logL, ax_occ, ax_theta, records: list[IterRecord]) -> None:
    """Trace plots over the narrated range: logL, state occupancy, Θ."""
    its = [r.iter_idx for r in records]

    ax_logL.plot(its, [r.logL_after for r in records], "o-", lw=1.4, color="C0")
    ax_logL.set_xlabel("Gibbs iter"); ax_logL.set_ylabel("complete-data logL")
    ax_logL.set_title("logL trace"); ax_logL.grid(alpha=0.3)

    occ = np.array([[ (r.z_ev_after == k).mean() for k in range(K)] for r in records])
    for k, c in zip(range(K), Z_EV_STATE_COLORS):
        ax_occ.plot(its, occ[:, k], "o-", lw=1.4, color=c, label=STATE_NAMES[k])
    ax_occ.set_xlabel("Gibbs iter"); ax_occ.set_ylabel("fraction of cells")
    ax_occ.set_title("z^EV state occupancy"); ax_occ.legend(fontsize=8); ax_occ.grid(alpha=0.3)

    ax_theta.plot(its, [r.theta_after[1] for r in records], "o-",
                  lw=1.4, color="C1", label=r"$\Theta_{low}$")
    ax_theta.plot(its, [r.theta_after[2] for r in records], "o-",
                  lw=1.4, color="C3", label=r"$\Theta_{high}$")
    ax_theta.axhspan(*THETA_BOUNDS[1], alpha=0.05, color="C1")
    ax_theta.axhspan(THETA_BOUNDS[2][0], min(THETA_BOUNDS[2][1], 8.0),
                     alpha=0.05, color="C3")
    ax_theta.set_xlabel("Gibbs iter"); ax_theta.set_ylabel("Θ (kW)")
    ax_theta.set_title("Θ trace"); ax_theta.legend(fontsize=8); ax_theta.grid(alpha=0.3)


# ===========================================================================
# Section 5.  Narrate one iteration
# ===========================================================================

def narrate_iteration(
    record:         IterRecord,
    z_true:         np.ndarray,    # (D, T)
    home_x:         np.ndarray,    # (D, T)
    x_nev_true:     np.ndarray,    # (D, T)
    x_ev_true:      np.ndarray,    # (D, T)
    days_to_show:   Sequence[int],
) -> None:
    """Print + plot one full C=1 Gibbs sweep (Blocks A, B, C). Call once per iter."""
    it = record.iter_idx
    print(f"\n{'═' * 78}")
    print(f"  Iteration {it}   (C = 1 fixed)")
    print(f"{'═' * 78}")

    # ── Block A: z^EV ───────────────────────────────────────────────────────
    print("\n  ── Block A: sample z^EV | Θ, z^LDS  (HMM FFBS) ──────────────────────")
    frac_b = _state_fractions(record.z_ev_before)
    frac_a = _state_fractions(record.z_ev_after)
    print(f"    state freq before : off={frac_b[0]:.3f} low={frac_b[1]:.3f} high={frac_b[2]:.3f}")
    print(f"    state freq after  : off={frac_a[0]:.3f} low={frac_a[1]:.3f} high={frac_a[2]:.3f}")
    n_flipped = int((record.z_ev_before != record.z_ev_after).sum())
    print(f"    cells changed     : {n_flipped:,} / {record.z_ev_after.size:,}")

    n_days = len(days_to_show)
    fig = plt.figure(figsize=(14, 0.6 * max(n_days, 1) + 1.5))
    plot_z_ev_per_day_grid(fig, z_true, record.z_ev_after,
                           record.z_ev_marginals, days_to_show, iter_idx=it)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()

    # ── Block B: Θ ────────────────────────────────────────────────────────
    print("\n  ── Block B: sample Θ_k for k ∈ {low, high} ──────────────────────────")
    for k in (1, 2):
        post = record.theta_posterior[k]
        suffix = "  (no z=k cells — sampled from prior)" if post["used_prior_only"] else ""
        print(f"    Θ_{STATE_NAMES[k]:>4}  : "
              f"m={post['m']:+.4f}  sd={post['sd']:.4f}  "
              f"trunc=[{post['lb']}, {'∞' if not np.isfinite(post['ub']) else post['ub']}]  "
              f"n_cells={post['n_cells']:>5}   →  drawn={record.theta_after[k]:+.4f}{suffix}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 3))
    plot_theta_posterior(axes[0], record, k=1)
    plot_theta_posterior(axes[1], record, k=2)
    plt.tight_layout(); plt.show()

    # ── Block C: z^LDS ───────────────────────────────────────────────────
    print("\n  ── Block C: sample z^LDS | z^EV, Θ  (Kalman FFBS) ───────────────────")
    delta = record.nonev_mean_after - record.nonev_mean_before
    print(f"    mean |Δ(C z^LDS)| across (d,t) = {np.abs(delta).mean():.4f} kW")
    print(f"    std of after - before          = {delta.std():.4f} kW")

    fig, ax = plt.subplots(figsize=(15, 3.5))
    plot_z_lds_compare(ax, home_x, record.nonev_mean_before,
                       record.nonev_mean_after, days_to_show,
                       title=f"z^LDS update (iter {it})")
    plt.tight_layout(); plt.show()

    # ── End-of-iter power decomposition ───────────────────────────────────
    print(f"\n  ── End of iter {it}:  complete-data logL = {record.logL_after:+.2f}")
    print("    Power decomposition (predicted vs true) across the chosen days:")
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    plot_power_decomposition(
        axes, home_x=home_x, x_nev_true=x_nev_true, x_ev_true=x_ev_true,
        nonev_pred=record.nonev_mean_after, z_ev=record.z_ev_after,
        theta=record.theta_after, days_to_show=days_to_show, iter_idx=it,
    )
    plt.tight_layout(); plt.show()


def _state_fractions(z: np.ndarray) -> np.ndarray:
    """(K,) fraction of cells in each EV state."""
    return np.array([(z == k).mean() for k in range(K)])


# ===========================================================================
# Section 6.  Top-level driver
# ===========================================================================

def run_diagnostic(
    home_x:         np.ndarray,
    params:         ModelParams,
    *,
    silent_iters:   int,
    narrate_iters:  int,
    seed:           int = 0,
) -> tuple[list[IterRecord], ChainState]:
    """Run silent_iters → start narrating for narrate_iters C=1 sweeps. Return records."""
    rng    = np.random.default_rng(seed)
    state  = init_state(home_x, params)
    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    print(f"  Running C=1 silent for {silent_iters} iterations …")
    run_silent(state, home_x, params, log_pi, log_P, rng, silent_iters)
    print(f"  Done. Now narrating iterations {silent_iters} … "
          f"{silent_iters + narrate_iters - 1}")

    records: list[IterRecord] = []
    for j in range(narrate_iters):
        rec = step_one_iter(state, home_x, params, log_pi, log_P, rng,
                             iter_idx=silent_iters + j)
        records.append(rec)
    return records, state
