"""Presentation plots for the two-track inference notebook (infer-model.ipynb).

These operate on the *results* of inference — `HomeResult` objects from
`gm.infer_all`, the heuristic baseline states, and the labelled test
dataframe — as opposed to `gibbs_diagnostics.py`, which steps through a single
C=1 Gibbs sweep. The EV-state colour palette (purple=off, teal=low,
yellow=high) and the discrete / fuzzy-blend RGB mappers are reused from there
so every z^EV picture across both notebooks shares one colour language.

Contents:
    Section 0   data + evidence helpers   — per-home arrays, axis ticks,
                                             C-decision selectors
    Section 1   C confusion (counts)
    Section 2   z confusion (row-normalised recall)
    Section 3   EV charging magnitudes — estimated Θ vs empirical spread
    Section 4   z^EV carpet heatmaps    — days × time-of-day, four variants
    Section 5   Gibbs convergence diagnostics
    Section 6   power decomposition over consecutive days (Non-EV / EV)
    Section 7   model-evidence scatter   — log p(x,C=1) vs log p(x,C=0)
    Section 8   background heatmaps      — true non-EV load vs inferred shape
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gibbs_diagnostics import (
    Z_EV_STATE_COLORS, z_ev_discrete_rgb, z_ev_blend_rgb,
)
from models.graphical_model.params import K, STATE_NAMES, T


# ===========================================================================
# Section 0.  Data + evidence helpers
# ===========================================================================

def home_arrays(df: pd.DataFrame, hid: int) -> dict:
    """Pull one home's labelled signals into (D, T) arrays.

    Returns total/ev/non_ev power, the true charge state, the ordered list of
    calendar dates (one per day-row), and the EV-ownership flag.
    """
    g = df[df["home_id"] == hid].sort_values(["day", "time_index"])
    D = g["day"].nunique()
    days = (g["day"].drop_duplicates().sort_values().reset_index(drop=True))
    return dict(
        D       = D,
        total   = g["total_load"].to_numpy().reshape(D, T).astype(float),
        ev      = g["ev_load"].to_numpy().reshape(D, T).astype(float),
        non_ev  = g["non_ev_load"].to_numpy().reshape(D, T).astype(float),
        z_true  = g["charge_state"].to_numpy().reshape(D, T),
        days    = days,                       # (D,) tz-aware datetimes (UTC)
        has_ev  = bool(g["has_ev"].iloc[0]),
    )


def _time_ticks(n: int = 5) -> tuple[np.ndarray, list[str]]:
    """`n` evenly-spaced time-of-day ticks: positions in [0, T) and HH:MM labels (UTC)."""
    idx = np.linspace(0, T - 1, n).round().astype(int)
    labels = [f"{(i * 15) // 60:02d}:{(i * 15) % 60:02d}" for i in idx]
    return idx, labels


def _date_ticks(days: pd.Series, n: int = 5) -> tuple[np.ndarray, list[str]]:
    """`n` day-axis ticks (always including first and last) as MM-DD labels."""
    D = len(days)
    idx = np.unique(np.linspace(0, D - 1, n).round().astype(int))
    labels = [days.iloc[i].strftime("%m-%d") for i in idx]
    return idx, labels


def c1_log_evidence(c1, which: str = "rb") -> float:
    """Pick a C=1 log-evidence estimate from a HomeInferenceC1.

    'rb'     — z^LDS-marginal (A′; comparable to C=0, the default decision).
    'plugin' — complete-data plug-in joint (A; NOT comparable — see specs §5).
    'chib'   — exact Chib marginal (B; only present if compute_chib was set).
    """
    table = {"rb": c1.log_evidence_rb,
             "plugin": c1.log_evidence_plugin,
             "chib": c1.log_evidence_chib}
    val = table[which]
    if val is None:
        raise ValueError(f"C=1 evidence '{which}' was not computed for this home.")
    return float(val)


def two_track_c_probs(inferences: dict, which: str = "rb") -> dict[int, float]:
    """Per-home soft P(C=1 | x) = softmax(log p(x,C=1)[which], log p(x,C=0)).

    `which` selects which C=1 evidence estimate drives the decision, letting
    the notebook swap 'rb' / 'plugin' / 'chib' without re-running inference.
    """
    out: dict[int, float] = {}
    for hid, r in inferences.items():
        le0 = float(r.c0.log_evidence)
        le1 = c1_log_evidence(r.c1, which)
        m = max(le0, le1)
        w0, w1 = np.exp(le0 - m), np.exp(le1 - m)
        out[hid] = float(w1 / (w0 + w1))
    return out


def two_track_c_pred(inferences: dict, which: str = "rb",
                     threshold: float = 0.5) -> dict[int, int]:
    """Hard Ĉ per home from the chosen evidence (threshold on the soft prob)."""
    return {hid: int(p >= threshold)
            for hid, p in two_track_c_probs(inferences, which).items()}


def true_c_by_home(df: pd.DataFrame) -> dict[int, int]:
    """{home_id: 0/1} ground-truth EV ownership."""
    return {int(hid): int(g["has_ev"].iloc[0])
            for hid, g in df.groupby("home_id", sort=True)}


# ===========================================================================
# Section 1.  Confusion-matrix shading — green on correct, red on incorrect
# ===========================================================================
# Shared renderer for both the C and z confusion matrices. A cell is "correct"
# when its true state (row) equals its predicted state (column); correct cells
# are shaded green and incorrect cells red, with the *intensity* in [0, 1]
# driving darkness (0 → white, 1 → dark). A perfect classifier puts all mass on
# the diagonal at intensity 1, so every diagonal cell is dark green and the
# off-diagonal red cells fade to white.

_DIAGONALS_ = plt.get_cmap("Blues")
_OFF_DIAGONALS_ = plt.get_cmap("Reds")


def _render_confusion(ax, annot: np.ndarray, intensity: np.ndarray,
                      row_states: Sequence[int], n_cols: int,
                      col_labels: Sequence[str], row_labels: Sequence[str],
                      *, is_count: bool) -> None:
    """Draw a confusion matrix with green-correct / red-incorrect shading.

    `annot` holds the values printed in each cell (counts or recall rates);
    `intensity` ∈ [0, 1] (NaN → grey "undefined") drives the green/red darkness.
    A cell is correct when row_states[i] == column j.
    """
    n_rows = len(row_states)
    img = np.ones((n_rows, n_cols, 4))
    for i, rs in enumerate(row_states):
        for j in range(n_cols):
            v = intensity[i, j]
            if np.isnan(v):
                img[i, j] = (0.92, 0.92, 0.92, 1.0)        # undefined → grey
            else:
                cmap = _DIAGONALS_ if rs == j else _OFF_DIAGONALS_
                img[i, j] = cmap(float(np.clip(v, 0.0, 1.0)))
    ax.imshow(img, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            a, v = annot[i, j], intensity[i, j]
            if isinstance(a, float) and np.isnan(a):
                txt = "—"
            else:
                txt = f"{int(a)}" if is_count else f"{a:.2f}"
            dark = (not np.isnan(v)) and v > 0.55
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=14 if is_count else 12,
                    fontweight="bold" if is_count else "normal",
                    color="white" if dark else "black")

    ax.set_xticks(range(n_cols), col_labels)
    ax.set_yticks(range(n_rows), row_labels)


# ===========================================================================
# Section 1b.  C confusion matrix (absolute counts, precision shading)
# ===========================================================================

def plot_c_confusion(true_by_home: dict[int, int],
                     pred_by_home: dict[int, int],
                     title: str,
                     ax=None):
    """2×2 confusion of EV ownership in **absolute home counts**.

    Rows = true C, columns = predicted Ĉ. Cells are shaded green (correct,
    diagonal) / red (incorrect) with intensity = **precision** (column-
    normalised: cell / column total), so a perfect classifier makes both
    diagonal cells dark green. Accuracy is shown in the title.
    """
    labels = ["no-EV", "EV"]
    cm = np.zeros((2, 2), dtype=int)
    for hid, ct in true_by_home.items():
        if hid in pred_by_home:
            cm[ct, pred_by_home[hid]] += 1
    n = cm.sum()
    acc = np.trace(cm) / n if n else float("nan")

    # Precision normalisation: divide each column by its total (NaN if empty).
    col_tot = cm.sum(axis=0).astype(float)
    intensity = np.divide(cm, col_tot, out=np.full(cm.shape, np.nan),
                          where=col_tot > 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.4))
    _render_confusion(ax, cm.astype(float), intensity, [0, 1], 2,
                      labels, labels, is_count=True)
    ax.set_xlabel("predicted  Ĉ")
    ax.set_ylabel("true  C")
    ax.set_title(f"{title}\n({n} homes, accuracy = {acc:.2f})", fontsize=10)
    return ax


# ===========================================================================
# Section 2.  z confusion matrices (row-normalised recall)
# ===========================================================================

def plot_z_confusion(cm: np.ndarray,
                     row_states: Sequence[int],
                     title: str,
                     ax=None):
    """Render a z^EV confusion matrix of **row-normalised recall** (each true
    row sums to 1; NaN rows = no ground-truth cells of that state).

    `cm` has shape (len(row_states), K). For EV homes pass the full 3×K matrix
    with row_states=[0,1,2]; for non-EV homes pass the single off-row (1×K)
    with row_states=[0] — there is no low/high ground truth to recall.
    """
    cm = np.atleast_2d(cm)
    n_rows = len(row_states)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 1.1 * n_rows + 1.4))
    # Same green-correct / red-incorrect shading as the C matrix; here the
    # intensity is the recall value itself (rows already sum to 1), so the
    # diagonal darkens with better per-state recall.
    _render_confusion(ax, cm, cm, list(row_states), K,
                      STATE_NAMES, [STATE_NAMES[k] for k in row_states],
                      is_count=False)
    ax.set_xlabel("predicted  ẑ")
    ax.set_ylabel("true  z")
    ax.set_title(title, fontsize=10)
    return ax


# ===========================================================================
# Section 3.  EV charging magnitudes — estimated Θ vs empirical spread
# ===========================================================================

def plot_ev_magnitudes(df: pd.DataFrame,
                       inferences: dict,
                       params,
                       ev_home_ids: Sequence[int],
                       *,
                       n_sigma: float = 1.96,
                       max_pts: int = 400,
                       seed: int = 0,
                       ax=None):
    """Per EV home, the estimated charging magnitude vs the empirical truth.

    For each home and each charging state k ∈ {low, high}:
      • a marker at the per-home posterior mean Θ̂_k (`c1.theta_mean[k]`) with a
        ±`n_sigma`·σ_ev[k] band (the *fixed* emission std, √sigma2_ev[k]); and
      • just to its right, a jittered strip of the true `ev_load` values for
        cells whose **true** charge state is k.
    Homes occupy separate x-axis sections; low (teal) and high (yellow) share
    the plot.
    """
    rng = np.random.default_rng(seed)
    states = [1, 2]                       # low, high
    colors = {k: Z_EV_STATE_COLORS[k] for k in states}

    if ax is None:
        _, ax = plt.subplots(figsize=(2.4 * len(ev_home_ids) + 1.5, 4.2))

    group_w = 1.0                          # x-span allotted per home section
    centers = []                           # for per-home x labels
    for hi, hid in enumerate(ev_home_ids):
        arr = home_arrays(df, hid)
        c1 = inferences[hid].c1
        base = hi * (len(states) + 0.6)    # left edge of this home's section
        centers.append(base + 0.5 * (len(states) - 1))
        for si, k in enumerate(states):
            x0 = base + si

            # Estimated mean ± n_sigma · σ_ev  (the model's prediction).
            m = float(c1.theta_mean[k])
            sd = float(np.sqrt(params.sigma2_ev[k]))
            ax.errorbar(x0, m, yerr=n_sigma * sd, fmt="o", ms=8,
                        color=colors[k], mec="black", mew=0.8, capsize=5,
                        elinewidth=2, zorder=3)

            # Empirical true ev_load for cells in true state k (jittered right).
            pts = arr["ev"][arr["z_true"] == k]
            if pts.size > max_pts:
                pts = rng.choice(pts, size=max_pts, replace=False)
            jit = x0 + 0.30 + rng.uniform(-0.08, 0.08, size=pts.size)
            ax.scatter(jit, pts, s=8, color=colors[k], alpha=0.20,
                       edgecolors="none", zorder=2)

    # x labels: a state tick per cluster, a home label centred under each section.
    xticks, xlabels = [], []
    for hi, hid in enumerate(ev_home_ids):
        base = hi * (len(states) + 0.6)
        for si, k in enumerate(states):
            xticks.append(base + si)
            xlabels.append(STATE_NAMES[k])
    ax.set_xticks(xticks, xlabels, fontsize=9)
    for c, hid in zip(centers, ev_home_ids):
        ax.text(c, -0.14, f"home {hid}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold")

    # Section dividers.
    for hi in range(1, len(ev_home_ids)):
        ax.axvline(hi * (len(states) + 0.6) - 0.8, color="0.8", lw=1)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=colors[1], mec="black",
               ls="", label=f"estimate Θ̂ ± {n_sigma:.2f}σ"),
        Line2D([0], [0], marker="o", color="0.5", alpha=0.4, ls="",
               label="true ev_load (cells in state)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylabel("EV charging power (kW)")
    ax.set_title("Estimated charging magnitude vs. empirical EV load by state",
                 fontsize=11)
    ax.grid(alpha=0.2, axis="y")
    ax.margins(x=0.02)
    return ax


# ===========================================================================
# Section 4.  z^EV carpet heatmaps — days (x) × time-of-day (y)
# ===========================================================================

def _resolve_day_range(D: int, day_range: tuple[int, int] | None) -> tuple[int, int]:
    if day_range is None:
        return 0, D
    lo, hi = day_range
    return max(0, lo), min(D, hi)


def _carpet(ax, rgb_DT3: np.ndarray, days: pd.Series, title: str) -> None:
    """Render one (D, T, 3) image transposed to time (y, 00:00 top) × days (x)."""
    img = rgb_DT3.transpose(1, 0, 2)                  # (T, D, 3)
    ax.imshow(img, aspect="auto", interpolation="nearest", origin="upper")
    t_idx, t_lab = _time_ticks(5)
    d_idx, d_lab = _date_ticks(days, 5)
    ax.set_yticks(t_idx, t_lab)
    ax.set_xticks(d_idx, d_lab)
    ax.set_ylabel("time of day (UTC)")
    ax.set_xlabel("day")
    ax.set_title(title, fontsize=11)


def _fold_low_discrete(z: np.ndarray, hide_low: bool) -> np.ndarray:
    """Map low(1) → off(0) so only off/high remain, if hide_low is set."""
    if not hide_low:
        return z
    z = z.copy()
    z[z == 1] = 0
    return z


def _fold_low_marginals(marg: np.ndarray, hide_low: bool) -> np.ndarray:
    """Fold the low-state posterior mass into off, if hide_low is set."""
    if not hide_low:
        return marg
    marg = marg.copy()
    marg[..., 0] += marg[..., 1]      # low mass → off
    marg[..., 1] = 0.0
    return marg


def plot_z_carpets(df: pd.DataFrame,
                   inferences: dict,
                   heur_states: dict,
                   hid: int,
                   *,
                   day_range: tuple[int, int] | None = None,
                   hide_low: bool = False,
                   figsize: tuple[float, float] = (12.0, 3.2)):
    """Four SEPARATE wide carpet figures for one home (days × time-of-day):

        Truth (discrete) | Posterior MAP (discrete) |
        Posterior fuzzy (γ-blend) | Heuristic (discrete)

    All share the purple/teal/yellow state palette; the fuzzy panel blends the
    three colours by posterior mass. Returns the list of figures.

    `hide_low=True` collapses the low state into off (low cells render purple,
    and the fuzzy panel reassigns low mass to off), leaving only off vs high —
    useful when low charging is noise you don't want to present.
    """
    arr = home_arrays(df, hid)
    D = arr["D"]
    lo, hi = _resolve_day_range(D, day_range)
    days = arr["days"].iloc[lo:hi].reset_index(drop=True)
    inf = inferences[hid]

    z_true = _fold_low_discrete(arr["z_true"][lo:hi], hide_low)
    z_map  = _fold_low_discrete(inf.z_hat[lo:hi], hide_low)
    gamma  = _fold_low_marginals(inf.z_marginals[lo:hi], hide_low)
    heur_full = heur_states.get(hid, np.zeros(D * T, dtype=int))[: D * T].reshape(D, T)
    z_heur = _fold_low_discrete(heur_full[lo:hi], hide_low)

    suffix = "  [low hidden]" if hide_low else ""
    panels = [
        (z_ev_discrete_rgb(z_true), f"home {hid} — ground-truth z (days {lo}–{hi-1}){suffix}"),
        (z_ev_discrete_rgb(z_map),  f"home {hid} — posterior MAP ẑ{suffix}"),
        (z_ev_blend_rgb(gamma),     f"home {hid} — posterior γ (fuzzy blend){suffix}"),
        (z_ev_discrete_rgb(z_heur), f"home {hid} — heuristic baseline{suffix}"),
    ]
    figs = []
    for rgb, title in panels:
        fig, ax = plt.subplots(figsize=figsize)
        _carpet(ax, rgb, days, title)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ===========================================================================
# Section 5.  Gibbs convergence diagnostics
# ===========================================================================

def plot_convergence(c1, title: str | None = None):
    """logL / Θ / state-occupancy traces from a C=1 chain, burn-in marked."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(c1.loglik_trace, lw=1)
    axes[0].axvline(c1.S_burn, color="C3", alpha=0.5, label="end burn-in")
    axes[0].set_ylabel("log p(x | state)")
    axes[0].set_title(title or "C=1 Gibbs traces")
    axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3)

    for k, name, color in [(1, "low", "C1"), (2, "high", "C2")]:
        axes[1].plot(c1.theta_trace[:, k], lw=1, color=color,
                     label=fr"$\Theta_{{{name}}}$")
    axes[1].axvline(c1.S_burn, color="C3", alpha=0.5)
    axes[1].set_ylabel("kW"); axes[1].legend(loc="right"); axes[1].grid(alpha=0.3)

    for k, color in [(0, "C0"), (1, "C1"), (2, "C2")]:
        axes[2].plot(c1.state_occ_trace[:, k], lw=1, color=color,
                     label=STATE_NAMES[k])
    axes[2].axvline(c1.S_burn, color="C3", alpha=0.5)
    axes[2].set_ylabel("fraction of cells"); axes[2].set_xlabel("Gibbs iter")
    axes[2].legend(loc="right"); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ===========================================================================
# Section 6.  Power decomposition over consecutive days
# ===========================================================================

def plot_power_decomposition(df: pd.DataFrame,
                             inferences: dict,
                             hid: int,
                             *,
                             day_range: tuple[int, int] | None = None,
                             figsize: tuple[float, float] = (14.0, 6.0)):
    """Two stacked panels over consecutive days (dotted-grey day dividers):

        1. Non-EV power:  true `non_ev_load`  vs  predicted  E[C z^LDS | x]
        2. EV power:      true `ev_load`       vs  predicted  Θ̂[ẑ_MAP]

    Both predictions are the central (MAP / posterior-mean) estimate — "just
    the z". x is concatenated days with calendar-date dividers.
    """
    arr = home_arrays(df, hid)
    D = arr["D"]
    lo, hi = _resolve_day_range(D, day_range)
    days_sel = list(range(lo, hi))
    dates = arr["days"]
    c1 = inferences[hid].c1

    nonev_pred = c1.z_lds_mean                       # (D, T)  E[C z^LDS | x]
    ev_pred = c1.theta_mean[c1.z_hat]                # (D, T)  Θ̂[ẑ_MAP]

    seg = np.arange(T) * 15 / 60                      # 0..24 h within a day
    xs = np.concatenate([seg + 24 * i for i in range(len(days_sel))])
    offsets = [24 * i for i in range(len(days_sel))]

    def _cat(a2d):
        return np.concatenate([a2d[d] for d in days_sel])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    

    panels = [
        (axes[0], 
        "Non-EV power", _cat(arr["non_ev"]),
        #  _cat(nonev_pred), r"predicted $\mathbb{E}[C z^{LDS}\mid x]$"
        ),
        (axes[1], "EV power", _cat(arr["ev"]), 
        # _cat(ev_pred),
        # r"predicted $\hat\Theta[\hat z_{MAP}]$"
        ),
    ]
    # for ax, title, truth, pred, pred_lbl in panels:
    for ax, title, truth in panels:
        ax.plot(xs, truth, lw=1.2, color="C0", label="true")
        # ax.plot(xs, pred, lw=1.4, color="C3", label=pred_lbl)
        for off in offsets[1:]:
            ax.axvline(off, color="0.6", lw=1.2, ls=":")
        ax.set_ylabel("kW")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2)
    axes[1].set_xticks([o + 12 for o in offsets],
                       [dates.iloc[d].strftime("%m-%d") for d in days_sel],
                       fontsize=8)
    axes[1].set_xlabel("day")
    fig.suptitle(f"home {hid} — power decomposition (days {lo}–{hi-1})", fontsize=11)
    fig.tight_layout()
    return fig


# ===========================================================================
# Section 7.  Model-evidence scatter
# ===========================================================================

def plot_evidence_scatter(df: pd.DataFrame,
                          inferences: dict,
                          *,
                          which: str = "rb",
                          ax=None):
    """Square scatter of log p(x, C=1) (y) vs log p(x, C=0) (x), one dot per
    home, EV homes green and non-EV orange, with the decision diagonal y = x
    dashed. Points above the diagonal are classified EV.
    """
    true_c = true_c_by_home(df)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))

    pts = {0: ([], []), 1: ([], [])}                  # true_c -> (x=C0, y=C1)
    for hid, r in inferences.items():
        ct = true_c.get(hid, 0)
        pts[ct][0].append(float(r.c0.log_evidence))
        pts[ct][1].append(c1_log_evidence(r.c1, which))

    for ct, color, label in [(1, "orange", "EV home"), (0, "lightblue", "non-EV home")]:
        ax.scatter(pts[ct][0], pts[ct][1], s=55, color=color, alpha=0.85,
                   edgecolors="black", linewidths=0.6, label=label, zorder=3)

    all_x = pts[0][0] + pts[1][0]
    all_y = pts[0][1] + pts[1][1]
    lo = min(all_x + all_y); hi = max(all_x + all_y)
    pad = 0.03 * (hi - lo + 1e-9)
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.7, label="decision boundary  (y = x)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\log p(x,\, C{=}0)$")
    ax.set_ylabel(rf"$\log p(x,\, C{{=}}1)$   [{which}]")
    ax.set_title("Model evidence: C=1 vs C=0 per home", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    return ax


# ===========================================================================
# Section 8.  Background heatmaps — true non-EV load vs inferred shape
# ===========================================================================
# Side-by-side day × time-of-day power heatmaps in the style of the EDA
# `power_heatmaps.png` (time of day on y with 00:00 at top, calendar day on x,
# magma colour, one shared colorbar). Left panel is the home's measured non-EV
# load; right panel is the model's inferred background E[C z^LDS | x], so the
# two are directly comparable cell-for-cell on a single colour scale.

def _bg_heat(ax, mat_DT: np.ndarray, days: pd.Series, title: str,
             *, vmin: float, vmax: float, cmap: str):
    """Render one (D, T) power matrix as a time(y, 00:00 top) × day(x) heatmap."""
    im = ax.imshow(mat_DT.T, aspect="auto", origin="upper",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    t_idx, t_lab = _time_ticks(5)
    d_idx, d_lab = _date_ticks(days, 5)
    ax.set_yticks(t_idx, t_lab)
    ax.set_xticks(d_idx, d_lab)
    ax.set_xlabel("day")
    ax.set_title(title, fontsize=11)
    return im


def plot_background_heatmaps(df: pd.DataFrame,
                             inferences: dict,
                             hid: int,
                             *,
                             day_range: tuple[int, int] | None = None,
                             pct: float = 99.0,
                             cmap: str = "magma",
                             figsize: tuple[float, float] = (15.0, 5.0)):
    """Two day × time-of-day heatmaps for one home (EDA `power_heatmaps` style):

        left  — ground-truth non-EV load (`non_ev_load`)
        right — inferred background shape  E[C z^LDS | x]  (`c1.z_lds_mean`)

    Time of day runs down the y-axis (00:00 at top, UTC), calendar days run
    along x. Both panels share one robust colour scale (0 → `pct`-th percentile
    of the pooled values) and a single colorbar, so the inferred reconstruction
    is directly comparable to the truth. Returns the Figure.
    """
    arr = home_arrays(df, hid)
    D = arr["D"]
    lo, hi = _resolve_day_range(D, day_range)
    days = arr["days"].iloc[lo:hi].reset_index(drop=True)

    truth = arr["non_ev"][lo:hi]                       # (d, T)  measured non-EV
    infer = inferences[hid].c1.z_lds_mean[lo:hi]       # (d, T)  E[C z^LDS | x]

    # Shared, outlier-robust colour scale over both panels.
    pooled = np.concatenate([truth.ravel(), infer.ravel()])
    vmin = 0.0
    vmax = float(np.nanpercentile(pooled, pct))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    _bg_heat(axes[0], truth, days, "ground-truth non-EV load",
             vmin=vmin, vmax=vmax, cmap=cmap)
    im = _bg_heat(axes[1], infer, days, r"inferred non-EV load  $\mathbb{E}[z^{non-EV}\mid x]$",
                  vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_ylabel("time of day (UTC)")

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("power (kW)")
    fig.suptitle(f"home {hid} — background power: truth vs inferred  "
                 f"(days {lo}–{hi-1})", fontsize=12)
    return fig
