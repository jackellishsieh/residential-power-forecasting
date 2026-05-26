"""Per-iteration step-through diagnostics for the collapsed Gibbs sampler.

Used by `notebooks/jack/gibbs-diagnostic.ipynb`. The sampler in
`models.graphical_model.inference.infer_home_collapsed` runs three Gibbs blocks
per iteration and returns only the final state; this module re-implements one
iteration as a *stepper* that exposes the intermediate quantities (the C
posterior, the Theta sufficient statistics, the residual after Block 2, etc.)
so each step can be inspected by eye.

Layout:
    Section 1   IterRecord + step_one_iter   — the stepper
    Section 2   run_silent                   — fast inner loop, no recording
    Section 3   z_ev categorical colormap helpers
    Section 4   plotters
    Section 5   narrate_iteration            — prints + plots one iteration
    Section 6   plot_traces                  — across the narrated range
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from models import graphical_model as gm
from models.graphical_model import ev, non_ev_lds
from models.graphical_model.params import (
    K, STATE_NAMES, T, THETA_BOUNDS, THETA_VAR_FLOOR, ModelParams,
)
from models.graphical_model.inference import compute_loglik, compute_loglik_c0


# ===========================================================================
# Section 1.  IterRecord + step_one_iter
# ===========================================================================

@dataclass
class IterRecord:
    """All intermediate quantities produced by one collapsed Gibbs iteration."""
    iter_idx: int

    # Block 1: collapsed C  ────────────────────────────────────────────────
    log_Z0:          float                  # log p(x | C=0, z_lds_prev, Theta_prev)
    log_Z1:          float                  # log p(x | C=1, z_lds_prev, Theta_prev)
    log_prior_c0:    float                  # log (1 - p_C)
    log_prior_c1:    float                  # log p_C
    p_c1_posterior:  float                  # softmax over (log_w0, log_w1)
    c_before:        int
    c_drawn:         int

    # Block 1b: z_ev | C  ──────────────────────────────────────────────────
    z_ev_before:     np.ndarray             # (D, T)  state at start of iter
    z_ev_after:      np.ndarray             # (D, T)  newly sampled (all-off if C=0)

    # Block 2: Theta_k for k ∈ {low, high}  ────────────────────────────────
    theta_before:    np.ndarray             # (K,)
    theta_after:     np.ndarray             # (K,)
    theta_posterior: dict                   # {k: {'m': ..., 'sd': ..., 'lb': .., 'ub': ..,
                                            #      'n_cells': ..., 'S_r': .., 'S_inv_var': ..}}

    # Block 3: z_lds  ──────────────────────────────────────────────────────
    nonev_mean_before: np.ndarray           # (D, T)  C_lds @ z_lds_prev
    nonev_mean_after:  np.ndarray           # (D, T)  C_lds @ z_lds_new

    # End-of-iter scalars  ─────────────────────────────────────────────────
    logL_after:      float                  # using post-Block-2 (z_ev, Theta) and pre-Block-3 nonev_mean


@dataclass
class ChainState:
    """Mutable state of the chain between iterations."""
    z_lds:   np.ndarray            # (D, L)
    z_ev:    np.ndarray            # (D, T)  int
    theta:   np.ndarray            # (K,)
    c:       int


def init_state(home_x: np.ndarray, params: ModelParams) -> ChainState:
    """Cold-start: z_lds = smoother mean on home_x (treats x as Non-EV);
    z_ev = all-off; theta = prior mean; c = 0."""
    D, _ = home_x.shape
    z_lds = params.lds.smooth(home_x).z_smooth                  # (D, L)
    return ChainState(
        z_lds = z_lds,
        z_ev  = np.zeros((D, T), dtype=np.int64),
        theta = params.mu_theta.copy(),
        c     = 0,
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
    """Run one collapsed-Gibbs iteration on `state` IN PLACE and return a
    diagnostic record of all intermediate quantities.

    Mirrors `infer_home_collapsed`'s main loop but returns intermediates.
    """
    lds       = params.lds
    C_lds     = lds.C
    nonev_var = np.diag(lds.R).copy()       # (T,)

    # Snapshot of pre-iter state
    z_ev_before        = state.z_ev.copy()
    theta_before       = state.theta.copy()
    c_before           = state.c
    nonev_mean_before  = state.z_lds @ C_lds.T                   # (D, T)

    # ──────────────────────────────────────────────────────────────────────
    # Block 1: collapsed C
    # ──────────────────────────────────────────────────────────────────────
    log_f, log_Z1 = ev.hmm_forward(home_x, state.theta, nonev_mean_before,
                                    nonev_var, params, log_pi, log_P)
    log_Z0 = compute_loglik_c0(home_x, nonev_mean_before, nonev_var, params)

    log_prior_c1 = float(np.log(params.p_C       + 1e-300))
    log_prior_c0 = float(np.log(1.0 - params.p_C + 1e-300))
    log_w1 = log_prior_c1 + log_Z1
    log_w0 = log_prior_c0 + log_Z0
    p_c1 = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))
    c_drawn = int(rng.random() < p_c1)

    # Block 1b: z_ev | C
    z_ev_after = (ev.hmm_backward_sample(log_f, params, rng) if c_drawn == 1
                  else np.zeros_like(state.z_ev))

    # ──────────────────────────────────────────────────────────────────────
    # Block 2: Theta_k for k ∈ {low, high}
    # Compute the closed-form posterior (mean, sd) for each k *before*
    # drawing, so the diagnostic can show what we sampled from.
    # ──────────────────────────────────────────────────────────────────────
    theta_after     = state.theta.copy()
    theta_posterior: dict = {}
    for k in (1, 2):
        post = _theta_k_posterior(home_x, z_ev_after, nonev_mean_before,
                                   nonev_var, params, k)
        theta_after[k] = ev._truncnorm_sample(post['m'], post['sd'],
                                               post['lb'], post['ub'], rng)
        theta_posterior[k] = post

    # ──────────────────────────────────────────────────────────────────────
    # Block 3: z_lds
    # ──────────────────────────────────────────────────────────────────────
    z_lds_after = non_ev_lds.sample_z_lds(home_x, z_ev_after, theta_after,
                                           params.sigma2_ev, lds, rng)
    nonev_mean_after = z_lds_after @ C_lds.T                     # (D, T)

    # End-of-iter logL (consistent with infer_home_collapsed.loglik_trace[it])
    logL_after = compute_loglik(home_x, z_ev_after, theta_after,
                                 nonev_mean_before, nonev_var, params)

    # Commit new state
    state.z_lds  = z_lds_after
    state.z_ev   = z_ev_after
    state.theta  = theta_after
    state.c      = c_drawn

    return IterRecord(
        iter_idx          = iter_idx,
        log_Z0            = float(log_Z0),
        log_Z1            = float(log_Z1),
        log_prior_c0      = log_prior_c0,
        log_prior_c1      = log_prior_c1,
        p_c1_posterior    = p_c1,
        c_before          = c_before,
        c_drawn           = c_drawn,
        z_ev_before       = z_ev_before,
        z_ev_after        = z_ev_after,
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
    """Advance the chain by `n_iters` iterations without recording — used to
    fast-forward through burn-in before the narrated range."""
    lds       = params.lds
    C_lds     = lds.C
    nonev_var = np.diag(lds.R).copy()

    for _ in range(n_iters):
        nonev_mean = state.z_lds @ C_lds.T
        log_f, log_Z1 = ev.hmm_forward(home_x, state.theta, nonev_mean,
                                        nonev_var, params, log_pi, log_P)
        log_Z0 = compute_loglik_c0(home_x, nonev_mean, nonev_var, params)
        log_w1 = np.log(params.p_C       + 1e-300) + log_Z1
        log_w0 = np.log(1.0 - params.p_C + 1e-300) + log_Z0
        p_c1 = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))
        state.c = int(rng.random() < p_c1)
        state.z_ev = (ev.hmm_backward_sample(log_f, params, rng) if state.c == 1
                       else np.zeros_like(state.z_ev))
        for k in (1, 2):
            state.theta[k] = ev.sample_theta_k(home_x, state.z_ev, nonev_mean,
                                                nonev_var, params, k, rng)
        state.z_lds = non_ev_lds.sample_z_lds(home_x, state.z_ev, state.theta,
                                               params.sigma2_ev, lds, rng)


# ===========================================================================
# Section 3.  z_ev categorical colormap helpers
# ===========================================================================
# Six colors: (state ∈ {off, low, high}) × (correct, incorrect)
# Encoding: code = 2 * pred_state + (1 if pred != true else 0)
# Style: viridis-inspired (off=purple, low=teal, high=yellow).
# Convention: vivid = correct, pale/desaturated = incorrect (so wrong cells
# look "washed out" against the dominant correct colour).

Z_EV_COLORS = [
    "#440154",   # 0  off  correct   — viridis 0.0 (deep purple)
    "#b3a4c0",   # 1  off  incorrect — pale purple
    "#21918c",   # 2  low  correct   — viridis ~0.5 (teal)
    "#a0d3d0",   # 3  low  incorrect — pale teal
    "#fde725",   # 4  high correct   — viridis 1.0 (yellow)
    "#fcf2a5",   # 5  high incorrect — pale yellow
]
Z_EV_CMAP = ListedColormap(Z_EV_COLORS)
Z_EV_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), Z_EV_CMAP.N)
Z_EV_LABELS = [
    "off ✓", "off ✗", "low ✓", "low ✗", "high ✓", "high ✗",
]


def encode_z_ev_with_truth(z_pred: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    """Map (z_pred, z_true) per cell to a 0..5 code for Z_EV_CMAP."""
    correctness = (z_pred != z_true).astype(np.int64)
    return 2 * z_pred + correctness


# ===========================================================================
# Section 4.  Plotters
# ===========================================================================

def plot_z_ev_categorical(
    ax,
    z_pred: np.ndarray,
    z_true: np.ndarray,
    title:  str,
) -> None:
    """Full-D categorical heatmap of z_pred, tinted by correctness vs z_true.

    Used in the "global" version (one heatmap covers all days at once). For
    per-day strips, use `plot_z_ev_per_day_row` below.
    """
    code = encode_z_ev_with_truth(z_pred, z_true)
    D, T_ = code.shape
    ax.imshow(code, aspect="auto", cmap=Z_EV_CMAP, norm=Z_EV_NORM,
              extent=[0, 24, D, 0], interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("day")


def add_z_ev_legend(fig, y: float = -0.02) -> None:
    """Add the 6-class legend once per figure (not per axis)."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=lbl)
               for c, lbl in zip(Z_EV_COLORS, Z_EV_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, y), fontsize=9, frameon=False)


def plot_z_ev_per_day_panel(
    ax,
    z_pred_day: np.ndarray,   # (T,)
    z_true_day: np.ndarray,   # (T,)
    title:      str | None = None,
) -> None:
    """Single-day strip heatmap for z_pred[d] tinted by correctness vs z_true[d].

    Renders as a 1-row image (T cells wide) — call once per day, per variant.
    """
    code = encode_z_ev_with_truth(z_pred_day[None, :], z_true_day[None, :])
    ax.imshow(code, aspect="auto", cmap=Z_EV_CMAP, norm=Z_EV_NORM,
              extent=[0, 24, 0, 1], interpolation="nearest")
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.set_yticks([])


def plot_z_ev_per_day_grid(
    fig,
    z_true:        np.ndarray,    # (D, T)
    z_pred_before: np.ndarray,    # (D, T)
    z_pred_after:  np.ndarray,    # (D, T)
    days_to_show:  Sequence[int],
    iter_idx:      int,
) -> None:
    """Build a `(len(days), 3)` grid of per-day strip heatmaps: truth | before | after.

    Column titles only on top row; row label "day d" on the leftmost panel.
    Single shared legend at the bottom.
    """
    n = len(days_to_show)
    if n == 0:
        return
    axes = fig.subplots(n, 3, sharex=True, sharey=False, squeeze=False)
    col_titles = ["ground truth", f"before iter {iter_idx}", f"after iter {iter_idx}"]
    for i, d in enumerate(days_to_show):
        plot_z_ev_per_day_panel(axes[i, 0], z_true[d],        z_true[d])
        plot_z_ev_per_day_panel(axes[i, 1], z_pred_before[d], z_true[d])
        plot_z_ev_per_day_panel(axes[i, 2], z_pred_after[d],  z_true[d])
        axes[i, 0].set_ylabel(f"day {d}", fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=10)
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=10)
        if i == n - 1:
            for j in range(3):
                axes[i, j].set_xlabel("hour of day")
    add_z_ev_legend(fig, y=-0.04)


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
    """Three stacked plots over the chosen `days_to_show`:

        (1) Non-EV : ground-truth x^NonEV vs predicted (C z^LDS)
        (2) EV     : ground-truth x^EV    vs predicted Θ[z_ev]
        (3) Total  : ground-truth x       vs predicted (C z^LDS + Θ[z_ev])

    Days are concatenated along x with red separators and labels.
    """
    if len(days_to_show) == 0:
        return

    ev_pred = theta[z_ev]                                      # (D, T)
    seg_x   = np.arange(T) * 15 / 60                            # 0..24
    xs      = np.concatenate([seg_x + 24 * i for i in range(len(days_to_show))])
    offsets = [24 * i for i in range(len(days_to_show))]

    def _concat(arr2d):
        return np.concatenate([arr2d[d] for d in days_to_show])

    panels = [
        (axes[0], "Non-EV power",
         _concat(x_nev_true), _concat(nonev_pred)),
        (axes[1], "EV power",
         _concat(x_ev_true),  _concat(ev_pred)),
        (axes[2], "Total power",
         _concat(home_x),     _concat(nonev_pred + ev_pred)),
    ]
    for ax, title, truth, pred in panels:
        ax.plot(xs, truth, lw=1.2, color="C0", label="true")
        ax.plot(xs, pred,  lw=1.4, color="C3", label="predicted")
        for off in offsets[1:]:
            ax.axvline(off, color="red", lw=1.8, alpha=0.9)
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
    """Concatenate `days_to_show` along x-axis; show x, nonev_before, nonev_after.

    Vertical dotted lines separate days; day numbers are annotated at the top of
    each segment.
    """
    if len(days_to_show) == 0:
        ax.set_title(title + "  (no days selected)")
        return

    segments_x       = []
    segments_x_data  = []
    segments_b       = []
    segments_a       = []
    offsets          = [0]
    for d in days_to_show:
        seg_x = np.arange(T) * 15 / 60                    # hours of day, 0..24
        segments_x.append(seg_x + offsets[-1])
        segments_x_data.append(home_x[d])
        segments_b.append(nonev_before[d])
        segments_a.append(nonev_after[d])
        offsets.append(offsets[-1] + 24)                   # one "day" = 24 units on x

    xs = np.concatenate(segments_x)
    x_data    = np.concatenate(segments_x_data)
    nonev_b   = np.concatenate(segments_b)
    nonev_a   = np.concatenate(segments_a)

    ax.plot(xs, x_data,  lw=1.0, color="C0",        label=r"$x$ (observed)")
    ax.plot(xs, nonev_b, lw=1.4, color="C7", alpha=0.8,
            label=r"$\hat z^{LDS}$ before")
    ax.plot(xs, nonev_a, lw=1.4, color="C1",
            label=r"$\hat z^{LDS}$ after")

    for d, off in zip(days_to_show, offsets[:-1]):
        if off > 0:
            ax.axvline(off, color="red", lw=1.8, alpha=0.9)
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


def plot_theta_posterior(
    ax,
    record:        IterRecord,
    k:             int,
) -> None:
    """For state k, show: prior mean, posterior mean ± sd, truncation bounds, drawn value."""
    post   = record.theta_posterior[k]
    m      = post["m"]
    sd     = post["sd"]
    lb, ub = post["lb"], post["ub"]
    drawn  = record.theta_after[k]
    before = record.theta_before[k]

    # Plot a tiny "number line" — bounds, posterior, draws.
    ax.set_title(f"Θ_{STATE_NAMES[k]}  posterior  (n_cells={post['n_cells']})",
                  fontsize=10)
    # Truncation band
    ub_plot = min(ub, m + 4 * sd + 1) if np.isfinite(ub) else (m + 4 * sd + 1)
    ax.axvspan(lb, ub_plot, color="#eef", alpha=0.5,
                label=f"truncation [{lb}, {'∞' if not np.isfinite(ub) else ub}]")
    # Posterior pdf (unnormalised) sketch
    xs = np.linspace(max(lb - 0.5, m - 4*sd), min(ub_plot, m + 4*sd), 200)
    pdf = np.exp(-0.5 * ((xs - m) / sd) ** 2)
    pdf = pdf / pdf.max() * 0.8
    ax.plot(xs, pdf, color="C0", lw=1.5, label="posterior (truncated)")
    # Markers
    ax.axvline(m,      color="C0", lw=1.2, label=f"post. mean = {m:.3f}")
    ax.axvline(drawn,  color="C3", lw=1.8, label=f"drawn = {drawn:.3f}")
    ax.axvline(before, color="C7", lw=1.0, ls="--",
                label=f"before = {before:.3f}")
    ax.set_xlabel("Θ (kW)")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)


def plot_traces(
    ax_logL,
    ax_logZ,
    ax_theta,
    records: list[IterRecord],
) -> None:
    """Trace plots over the narrated range: logL, log_Z0/log_Z1, Θ."""
    its = [r.iter_idx for r in records]

    ax_logL.plot(its, [r.logL_after for r in records], "o-", lw=1.4, color="C0")
    ax_logL.set_xlabel("Gibbs iter"); ax_logL.set_ylabel("complete-data logL")
    ax_logL.set_title("logL trace")
    ax_logL.grid(alpha=0.3)

    ax_logZ.plot(its, [r.log_Z0 for r in records], "o-", lw=1.4, color="C7",
                 label=r"$\log Z_0$ (C=0)")
    ax_logZ.plot(its, [r.log_Z1 for r in records], "o-", lw=1.4, color="C3",
                 label=r"$\log Z_1$ (C=1)")
    ax_logZ.set_xlabel("Gibbs iter"); ax_logZ.set_ylabel("log-marginal")
    ax_logZ.set_title(r"$\log p(x | C=\cdot, z^{LDS}, \Theta)$")
    ax_logZ.legend(fontsize=8); ax_logZ.grid(alpha=0.3)

    ax_theta.plot(its, [r.theta_after[1] for r in records], "o-",
                  lw=1.4, color="C1", label=r"$\Theta_{low}$")
    ax_theta.plot(its, [r.theta_after[2] for r in records], "o-",
                  lw=1.4, color="C3", label=r"$\Theta_{high}$")
    ax_theta.axhspan(*THETA_BOUNDS[1], alpha=0.05, color="C1")
    ax_theta.axhspan(THETA_BOUNDS[2][0],
                     min(THETA_BOUNDS[2][1], 8.0),
                     alpha=0.05, color="C3")
    ax_theta.set_xlabel("Gibbs iter"); ax_theta.set_ylabel("Θ (kW)")
    ax_theta.set_title("Θ trace")
    ax_theta.legend(fontsize=8); ax_theta.grid(alpha=0.3)


# ===========================================================================
# Section 5.  Narrate one iteration
# ===========================================================================

def narrate_iteration(
    record:         IterRecord,
    z_true:         np.ndarray,    # (D, T)
    home_x:         np.ndarray,    # (D, T)
    x_nev_true:     np.ndarray,    # (D, T)   ground-truth Non-EV
    x_ev_true:      np.ndarray,    # (D, T)   ground-truth EV
    days_to_show:   Sequence[int],
) -> None:
    """Print + plot one full iteration. Call once per narrated iter.

    `days_to_show` is used identically for the z^EV per-day grid, the z^LDS
    update plot, and the end-of-iter power-decomposition plot so the same days
    can be tracked through all four blocks.
    """
    it = record.iter_idx
    print(f"\n{'═' * 78}")
    print(f"  Iteration {it}")
    print(f"{'═' * 78}")

    # ── Block 1: C ────────────────────────────────────────────────────────
    print("\n  ── Block 1: collapsed C ─────────────────────────────────────────────")
    print(f"    log p(x | C=0, z^LDS, Θ)         = {record.log_Z0:+12.2f}")
    print(f"    log p(x | C=1, z^LDS, Θ)         = {record.log_Z1:+12.2f}")
    print(f"    log prior log(1-p_C)/log(p_C)    = "
          f"{record.log_prior_c0:+7.3f}  /  {record.log_prior_c1:+7.3f}")
    print(f"    posterior P(C=1 | x, z^LDS, Θ)  = {record.p_c1_posterior:.6f}")
    print(f"    C_before = {record.c_before}     →     drawn C = {record.c_drawn}")

    # ── Block 1b: z_ev ────────────────────────────────────────────────────
    print("\n  ── Block 1b: sample z^EV | C ────────────────────────────────────────")
    frac_b = _state_fractions(record.z_ev_before)
    frac_a = _state_fractions(record.z_ev_after)
    print(f"    state freq before : off={frac_b[0]:.3f} "
          f"low={frac_b[1]:.3f} high={frac_b[2]:.3f}")
    print(f"    state freq after  : off={frac_a[0]:.3f} "
          f"low={frac_a[1]:.3f} high={frac_a[2]:.3f}")
    n_flipped = int((record.z_ev_before != record.z_ev_after).sum())
    print(f"    cells changed     : {n_flipped:,} / {record.z_ev_after.size:,}")

    n_days = len(days_to_show)
    fig = plt.figure(figsize=(14, 0.6 * max(n_days, 1) + 1.5))
    plot_z_ev_per_day_grid(fig, z_true,
                            record.z_ev_before, record.z_ev_after,
                            days_to_show, iter_idx=it)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()

    # ── Block 2: Θ ────────────────────────────────────────────────────────
    print("\n  ── Block 2: sample Θ_k for k ∈ {low, high} ──────────────────────────")
    for k in (1, 2):
        post = record.theta_posterior[k]
        suffix = "  (no z=k cells — sampled from prior)" if post["used_prior_only"] else ""
        print(f"    Θ_{STATE_NAMES[k]:>4}  : "
              f"m={post['m']:+.4f}  sd={post['sd']:.4f}  "
              f"trunc=[{post['lb']}, {'∞' if not np.isfinite(post['ub']) else post['ub']}]  "
              f"n_cells={post['n_cells']:>5}   →  drawn={record.theta_after[k]:+.4f}"
              f"{suffix}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 3))
    plot_theta_posterior(axes[0], record, k=1)
    plot_theta_posterior(axes[1], record, k=2)
    plt.tight_layout()
    plt.show()

    # ── Block 3: z_lds ────────────────────────────────────────────────────
    print("\n  ── Block 3: sample z^LDS | z^EV, Θ ──────────────────────────────────")
    delta = record.nonev_mean_after - record.nonev_mean_before
    print(f"    mean |Δ(C z^LDS)| across (d,t) = {np.abs(delta).mean():.4f} kW")
    print(f"    std of after - before          = {delta.std():.4f} kW")

    fig, ax = plt.subplots(figsize=(15, 3.5))
    plot_z_lds_compare(
        ax, home_x,
        record.nonev_mean_before, record.nonev_mean_after,
        days_to_show, title=f"z^LDS update (iter {it})",
    )
    plt.tight_layout()
    plt.show()

    # ── End-of-iter power decomposition ───────────────────────────────────
    print(f"\n  ── End of iter {it}:  complete-data logL = {record.logL_after:+.2f}")
    print("    Power decomposition (predicted vs true) across the chosen days:")

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    plot_power_decomposition(
        axes,
        home_x      = home_x,
        x_nev_true  = x_nev_true,
        x_ev_true   = x_ev_true,
        nonev_pred  = record.nonev_mean_after,    # (C z^LDS) post-Block-3
        z_ev        = record.z_ev_after,          # z^EV post-Block-1b
        theta       = record.theta_after,         # Θ post-Block-2
        days_to_show= days_to_show,
        iter_idx    = it,
    )
    plt.tight_layout()
    plt.show()


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
    """Run silent_iters → start narrating for narrate_iters. Return records.

    The notebook calls this, then iterates `records` for per-iter rendering.
    """
    rng    = np.random.default_rng(seed)
    state  = init_state(home_x, params)
    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    print(f"  Running silent for {silent_iters} iterations …")
    run_silent(state, home_x, params, log_pi, log_P, rng, silent_iters)
    print(f"  Done. Now narrating iterations {silent_iters} … "
          f"{silent_iters + narrate_iters - 1}")

    records: list[IterRecord] = []
    for j in range(narrate_iters):
        rec = step_one_iter(state, home_x, params, log_pi, log_P, rng,
                             iter_idx=silent_iters + j)
        records.append(rec)
    return records, state
