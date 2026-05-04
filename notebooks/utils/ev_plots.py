"""
EV charging magnitude visualization: empirical data vs model prediction.

Two public entry points:
  plot_ev_magnitude_comparison  --  stacked per-home plots (x = state, y = kW)
  plot_theta_posteriors         --  forest plot of per-home Theta posteriors vs prior

State color convention used throughout both figures:
  off  -> STATE_COLORS[0]  (grey)
  low  -> STATE_COLORS[1]  (cornflowerblue)
  high -> STATE_COLORS[2]  (tomato)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_NAMES  = ["off", "low", "high"]
STATE_COLORS = ["dimgray", "cornflowerblue", "tomato"]
STATE_MARKERS = ["s", "o", "^"]   # square / circle / triangle per state


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_theta_posterior(
    observed_ev_load_sum: float,      # S_y^(n) = sum of ev_load at state k
    num_observations: int,            # n^(n)_k = count of timesteps in state k
    mu_theta_k: float,                # prior mean
    sigma2_theta_k: float,            # prior variance
    sigma2_ev_k: float,               # per-observation noise variance
) -> tuple[float, float]:
    """Conjugate Gaussian posterior for one home's Theta_k.

    Model:
      Theta_k       ~ N(mu_theta_k, sigma2_theta_k)           [prior]
      ev_load[i]    ~ N(Theta_k,    sigma2_ev_k)              [likelihood]

    Returns (posterior_mean, posterior_std).
    """
    THETA_VAR_FLOOR = 1e-6
    effective_prior_variance = max(sigma2_theta_k, THETA_VAR_FLOOR)

    prior_precision = 1.0 / effective_prior_variance
    data_precision  = num_observations / sigma2_ev_k if num_observations > 0 else 0.0
    posterior_precision = prior_precision + data_precision

    prior_contribution = mu_theta_k / effective_prior_variance
    data_contribution  = observed_ev_load_sum / sigma2_ev_k if num_observations > 0 else 0.0
    posterior_mean = (prior_contribution + data_contribution) / posterior_precision

    return float(posterior_mean), float(np.sqrt(1.0 / posterior_precision))


def compute_home_ev_stats(home_df: pd.DataFrame, params) -> dict:
    """Compute empirical + model stats for one home's EV charging magnitudes.

    Returned dict has one entry per state k in {0, 1, 2}:
      'off', 'low', 'high' each containing:
        observations          np.ndarray   raw ev_load values at that state
        empirical_mean        float
        empirical_std         float
        posterior_theta_mean  float
        posterior_theta_std   float
        theta_only_lo/hi      float        posterior mean ± posterior std
        combined_lo/hi        float        posterior mean ± sqrt(posterior_std² + sigma_ev_k²)
    """
    home_id = int(home_df["home_id"].iloc[0])
    has_ev  = bool(home_df["has_ev"].iloc[0])

    per_state_stats = {}

    for k, state_name in enumerate(STATE_NAMES):
        state_mask        = home_df["charge_state"] == k
        ev_load_at_state  = home_df.loc[state_mask, "ev_load"].to_numpy(dtype=np.float64)
        num_observations  = len(ev_load_at_state)

        empirical_mean = float(ev_load_at_state.mean()) if num_observations > 0 else float(params.mu_theta[k])
        empirical_std  = float(ev_load_at_state.std(ddof=1)) if num_observations > 1 else 0.0

        posterior_theta_mean, posterior_theta_std = _compute_theta_posterior(
            observed_ev_load_sum = float(ev_load_at_state.sum()),
            num_observations     = num_observations,
            mu_theta_k           = float(params.mu_theta[k]),
            sigma2_theta_k       = float(params.sigma2_theta[k]),
            sigma2_ev_k          = float(params.sigma2_ev[k]),
        )

        sigma_ev_k   = float(np.sqrt(params.sigma2_ev[k]))
        combined_std = float(np.sqrt(posterior_theta_std ** 2 + params.sigma2_ev[k]))

        per_state_stats[state_name] = dict(
            observations         = ev_load_at_state,
            num_observations     = num_observations,
            empirical_mean       = empirical_mean,
            empirical_std        = empirical_std,
            posterior_theta_mean = posterior_theta_mean,
            posterior_theta_std  = posterior_theta_std,
            theta_only_lo        = posterior_theta_mean - posterior_theta_std,
            theta_only_hi        = posterior_theta_mean + posterior_theta_std,
            combined_lo          = posterior_theta_mean - combined_std,
            combined_hi          = posterior_theta_mean + combined_std,
        )

    return dict(
        home_id       = home_id,
        has_ev        = has_ev,
        per_state     = per_state_stats,
    )


def compute_prior_ev_stats(params) -> dict:
    """Stats for the prior (no home-specific data), one entry per state."""
    per_state_stats = {}
    for k, state_name in enumerate(STATE_NAMES):
        mu    = float(params.mu_theta[k])
        sigma = float(np.sqrt(params.sigma2_theta[k]))
        combined_std = float(np.sqrt(params.sigma2_theta[k] + params.sigma2_ev[k]))
        per_state_stats[state_name] = dict(
            posterior_theta_mean = mu,
            posterior_theta_std  = sigma,
            theta_only_lo        = mu - sigma,
            theta_only_hi        = mu + sigma,
            combined_lo          = mu - combined_std,
            combined_hi          = mu + combined_std,
        )
    return dict(per_state=per_state_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Single-axes drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_ev_state_on_axes(
    ax: Axes,
    x_center: float,            # horizontal position for this state
    state_stats: dict,
    state_color: str,
    state_marker: str,
    state_name: str,
    show_individual_points: bool,
    show_empirical_band: bool,
    show_theta_only_band: bool,
    show_combined_band: bool,
    max_scatter_points: int,
    jitter_width: float = 0.12,
) -> None:
    """Draw one state's layers at a given x position."""

    observations = state_stats.get("observations", None)
    posterior_mean = state_stats["posterior_theta_mean"]

    # ── true-data layers ──────────────────────────────────────────────────────
    if show_individual_points and observations is not None and len(observations) > 0:
        num_to_plot   = min(max_scatter_points, len(observations))
        sampled_obs   = np.random.default_rng(42).choice(observations, num_to_plot, replace=False)
        jitter        = np.random.default_rng(42).uniform(-jitter_width, jitter_width, num_to_plot)
        ax.scatter(
            x_center + jitter, sampled_obs,
            color=state_color, alpha=0.18, s=4, linewidths=0, zorder=2,
        )

    if show_empirical_band and observations is not None and len(observations) > 1:
        ax.errorbar(
            x_center - 0.18,
            state_stats["empirical_mean"],
            yerr=state_stats["empirical_std"],
            fmt=state_marker, color=state_color,
            capsize=5, capthick=1.2, elinewidth=1.2, ms=6, alpha=0.9,
            zorder=4,
            label=f"{state_name} empirical mean ± 1σ",
        )

    # ── model layers ──────────────────────────────────────────────────────────
    # Combined band (wider, drawn first so theta band appears on top)
    if show_combined_band:
        ax.errorbar(
            x_center + 0.0,
            posterior_mean,
            yerr=[[posterior_mean - state_stats["combined_lo"]],
                  [state_stats["combined_hi"] - posterior_mean]],
            fmt="none", color=state_color,
            elinewidth=6, alpha=0.25, capsize=0, zorder=3,
        )

    # Theta-only band (narrower, on top)
    if show_theta_only_band:
        ax.errorbar(
            x_center + 0.0,
            posterior_mean,
            yerr=[[posterior_mean - state_stats["theta_only_lo"]],
                  [state_stats["theta_only_hi"] - posterior_mean]],
            fmt="none", color=state_color,
            elinewidth=6, alpha=0.55, capsize=0, zorder=3,
        )

    # Posterior mean marker (always shown)
    ax.scatter(
        x_center, posterior_mean,
        marker=state_marker, color=state_color, s=50, zorder=5,
        edgecolors="white", linewidths=0.8,
    )


def _draw_home_ev_on_axes(
    ax: Axes,
    home_stats: dict,
    show_individual_points: bool,
    show_empirical_band: bool,
    show_theta_only_band: bool,
    show_combined_band: bool,
    max_scatter_points: int,
    title: str,
    ylim: tuple[float, float] | None,
) -> None:
    """Draw all three states for one home onto axes."""

    x_positions = {state_name: i for i, state_name in enumerate(STATE_NAMES)}

    for k, state_name in enumerate(STATE_NAMES):
        state_stats  = home_stats["per_state"][state_name]
        state_color  = STATE_COLORS[k]
        state_marker = STATE_MARKERS[k]
        _draw_ev_state_on_axes(
            ax                     = ax,
            x_center               = float(x_positions[state_name]),
            state_stats            = state_stats,
            state_color            = state_color,
            state_marker           = state_marker,
            state_name             = state_name,
            show_individual_points = show_individual_points,
            show_empirical_band    = show_empirical_band,
            show_theta_only_band   = show_theta_only_band,
            show_combined_band     = show_combined_band,
            max_scatter_points     = max_scatter_points,
        )

    # off state note: Theta fixed at 0 by construction
    ax.annotate(
        "fixed at 0", xy=(0, 0.02), xycoords=("data", "axes fraction"),
        fontsize=6, color="dimgray", ha="center",
    )

    ax.set_xticks(list(range(len(STATE_NAMES))))
    ax.set_xticklabels(STATE_NAMES, fontsize=9)
    ax.set_xlim(-0.6, len(STATE_NAMES) - 0.4)
    ax.set_title(title, fontsize=9, loc="left", pad=3)
    ax.set_ylabel("EV load (kW)", fontsize=8)
    ax.grid(axis="y", lw=0.35, alpha=0.4)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Build a compact legend (one entry per layer type, not per state)
    legend_handles = []
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    if show_empirical_band:
        legend_handles.append(
            mlines.Line2D([], [], color="k", marker="o", ls="none", ms=5,
                          label="empirical mean ± 1σ  (offset left)")
        )
    if show_theta_only_band:
        legend_handles.append(
            mpatches.Patch(color="k", alpha=0.55, label="Θ posterior ± 1σ")
        )
    if show_combined_band:
        legend_handles.append(
            mpatches.Patch(color="k", alpha=0.25, label="Θ + σ^EV combined ± 1σ")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=6, loc="upper left", framealpha=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Public: stacked per-home plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_ev_magnitude_comparison(
    train_df: pd.DataFrame,
    params,
    home_ids: list[int],
    *,
    show_individual_points: bool = True,
    show_empirical_band: bool = True,
    show_theta_only_band: bool = True,
    show_combined_band: bool = True,
    max_scatter_points: int = 300,
    row_height: float = 3.5,
    figure_width: float = 6.0,
) -> Figure:
    """Stacked single-column figure: one row per EV home (non-EV homes skipped).

    x-axis  : EV charging state  {off, low, high}
    y-axis  : EV load in kW,  shared absolute scale across all rows.

    Parameters
    ----------
    home_ids               : list of home_id values to include.
    show_individual_points : jittered scatter of raw ev_load observations.
    show_empirical_band    : empirical mean ± 1σ error bar (offset left of center).
    show_theta_only_band   : posterior Θ ± posterior std (thick CI bar).
    show_combined_band     : posterior Θ ± sqrt(posterior_std² + sigma_EV²).
    max_scatter_points     : max observations plotted per state (random downsample).
    """
    # Keep only EV homes
    ev_home_ids = [
        hid for hid in home_ids
        if bool(train_df[train_df["home_id"] == hid]["has_ev"].iloc[0])
    ]
    if not ev_home_ids:
        raise ValueError("No EV homes in home_ids; this plot requires EV homes.")

    num_rows = len(ev_home_ids)

    # ── compute all stats ─────────────────────────────────────────────────────
    all_home_stats = {
        hid: compute_home_ev_stats(train_df[train_df["home_id"] == hid], params)
        for hid in ev_home_ids
    }

    # ── shared y-axis limits ──────────────────────────────────────────────────
    all_upper_values = []
    for stats in all_home_stats.values():
        for state_name in STATE_NAMES:
            s = stats["per_state"][state_name]
            all_upper_values.append(s["combined_hi"])
            if s.get("observations") is not None and len(s["observations"]) > 0:
                all_upper_values.append(float(np.percentile(s["observations"], 99)))
    shared_ylim = (0.0, max(all_upper_values) * 1.10)

    # ── build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        num_rows, 1,
        figsize=(figure_width, row_height * num_rows),
        sharey=True,
        squeeze=False,
    )

    for row_index, hid in enumerate(ev_home_ids):
        stats   = all_home_stats[hid]
        num_obs = {s: stats["per_state"][s]["num_observations"] for s in STATE_NAMES}
        title   = (
            f"Home {hid}  "
            f"  n_off={num_obs['off']:,}  n_low={num_obs['low']:,}  n_high={num_obs['high']:,}"
        )
        is_last_row = (row_index == num_rows - 1)
        _draw_home_ev_on_axes(
            ax                     = axes[row_index, 0],
            home_stats             = stats,
            show_individual_points = show_individual_points,
            show_empirical_band    = show_empirical_band,
            show_theta_only_band   = show_theta_only_band,
            show_combined_band     = show_combined_band,
            max_scatter_points     = max_scatter_points,
            title                  = title,
            ylim                   = shared_ylim,
        )
        if not is_last_row:
            axes[row_index, 0].set_xlabel("")

    fig.suptitle("EV charging magnitude: empirical vs model  (EV homes only)", fontsize=11, y=1.002)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: theta posterior forest plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_theta_posteriors(
    train_df: pd.DataFrame,
    params,
    home_ids: list[int],
    *,
    figure_width: float = 11.0,
    figure_height: float = 5.5,
) -> Figure:
    """Forest plot: posterior Theta_k ± 1σ for each home, plus the prior at x=0.

    x-axis  : x=0 is the global prior; x=1, 2, ... are individual homes.
    y-axis  : EV load in kW.
    Colors  : one color per state (off=grey, low=blue, high=red).
    At each x, three state values are drawn as vertically-stacked colored dots
    connected by a thin vertical line to make per-home patterns readable.

    Non-EV homes are included as x-positions but shown with a note (Theta ≡ 0).
    """
    num_homes = len(home_ids)

    # ── compute posteriors ────────────────────────────────────────────────────
    posterior_means = {state: [] for state in STATE_NAMES}
    posterior_stds  = {state: [] for state in STATE_NAMES}
    home_is_ev      = []

    for hid in home_ids:
        home_df   = train_df[train_df["home_id"] == hid]
        is_ev     = bool(home_df["has_ev"].iloc[0])
        home_is_ev.append(is_ev)
        if is_ev:
            stats = compute_home_ev_stats(home_df, params)
            for state_name in STATE_NAMES:
                posterior_means[state_name].append(stats["per_state"][state_name]["posterior_theta_mean"])
                posterior_stds[state_name].append(stats["per_state"][state_name]["posterior_theta_std"])
        else:
            # Non-EV homes: Theta fixed at 0 for all states
            for state_name in STATE_NAMES:
                posterior_means[state_name].append(0.0)
                posterior_stds[state_name].append(0.0)

    # ── x positions ───────────────────────────────────────────────────────────
    x_prior = 0
    x_homes = list(range(1, num_homes + 1))
    x_all   = [x_prior] + x_homes
    x_tick_labels = ["prior"] + [
        f"{hid}{'*' if is_ev else ''}"
        for hid, is_ev in zip(home_ids, home_is_ev)
    ]

    prior_stats = compute_prior_ev_stats(params)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # ── draw prior at x=0 ─────────────────────────────────────────────────────
    for k, state_name in enumerate(STATE_NAMES):
        prior_s = prior_stats["per_state"][state_name]
        prior_mean = prior_s["posterior_theta_mean"]
        prior_std  = prior_s["posterior_theta_std"]
        combined_std = float(np.sqrt(params.sigma2_theta[k] + params.sigma2_ev[k]))

        # Combined CI (wide, faint)
        ax.errorbar(
            x_prior, prior_mean,
            yerr=combined_std,
            fmt="none", color=STATE_COLORS[k],
            elinewidth=8, alpha=0.18, capsize=0, zorder=2,
        )
        # Theta-only CI
        ax.errorbar(
            x_prior, prior_mean,
            yerr=prior_std,
            fmt="none", color=STATE_COLORS[k],
            elinewidth=8, alpha=0.45, capsize=0, zorder=3,
        )
        # Mean marker
        ax.scatter(
            x_prior, prior_mean,
            marker=STATE_MARKERS[k], color=STATE_COLORS[k],
            s=70, zorder=5, edgecolors="white", linewidths=1.0,
            label=f"{state_name}",
        )

    # Vertical connector for prior
    prior_y_values = [prior_stats["per_state"][s]["posterior_theta_mean"] for s in STATE_NAMES]
    ax.vlines(x_prior, min(prior_y_values), max(prior_y_values),
              color="dimgray", lw=0.8, alpha=0.5, zorder=1)

    # ── draw homes ────────────────────────────────────────────────────────────
    for x_pos, hid, is_ev in zip(x_homes, home_ids, home_is_ev):

        home_y_values = []
        for k, state_name in enumerate(STATE_NAMES):
            post_mean = posterior_means[state_name][x_pos - 1]
            post_std  = posterior_stds[state_name][x_pos - 1]
            combined_std = float(np.sqrt(post_std ** 2 + params.sigma2_ev[k]))

            if is_ev and k > 0:   # skip off-state CI for EV (trivially 0); skip all for non-EV
                ax.errorbar(
                    x_pos, post_mean, yerr=combined_std,
                    fmt="none", color=STATE_COLORS[k],
                    elinewidth=5, alpha=0.18, capsize=0, zorder=2,
                )
                ax.errorbar(
                    x_pos, post_mean, yerr=post_std,
                    fmt="none", color=STATE_COLORS[k],
                    elinewidth=5, alpha=0.45, capsize=0, zorder=3,
                )

            ax.scatter(
                x_pos, post_mean,
                marker=STATE_MARKERS[k], color=STATE_COLORS[k],
                s=55 if is_ev else 25,
                alpha=1.0 if is_ev else 0.35,
                zorder=5, edgecolors="white", linewidths=0.8,
            )
            home_y_values.append(post_mean)

        # Vertical connector across three state values for this home
        ax.vlines(x_pos, min(home_y_values), max(home_y_values),
                  color="dimgray", lw=0.7, alpha=0.4, zorder=1)

    # ── formatting ────────────────────────────────────────────────────────────
    ax.set_xticks(x_all)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Θ_k  (kW)", fontsize=9)
    ax.set_xlabel("Home ID  (* = EV home)  |  x=0 is the global prior\n"
                  "non-EV homes shown faded (Θ ≡ 0 by construction)", fontsize=8)
    ax.set_title(
        "Per-home posterior Θ_k ± 1σ  vs  global prior\n"
        "Thick bars = Θ-only CI;  faint bars = Θ + σ^EV combined CI;  "
        "dots connected by vertical line per home",
        fontsize=9,
    )
    ax.legend(
        title="state", fontsize=8, title_fontsize=8,
        loc="upper right", framealpha=0.8,
    )
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.grid(axis="x", lw=0.25, alpha=0.25)

    plt.tight_layout()
    return fig
