"""
Background load visualization: empirical data vs model prediction.

Two public entry points:
  plot_background_comparison  --  N+1 stacked time-series (one per home + prior)
  plot_alpha_posteriors       --  forest plot of per-home alpha posteriors vs prior

All plots sharing the time axis use hourly HH:MM labels.
All per-home time-series share the same absolute kW y-axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ─────────────────────────────────────────────────────────────────────────────
# Axis utilities
# ─────────────────────────────────────────────────────────────────────────────

def _apply_hourly_time_axis(ax: Axes) -> None:
    """Label x-axis in HH:MM at hourly (every 4th 15-min interval) ticks."""
    tick_positions = list(range(0, 96, 4))
    tick_labels = [
        f"{(i * 15) // 60:02d}:{(i * 15) % 60:02d}" for i in tick_positions
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, 95.5)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_alpha_posterior(
    empirical_day_mean: np.ndarray,  # (T,) per-home mean of non_ev_load across days
    num_days: int,
    rho: np.ndarray,                 # (T,) from ModelParams
    sigma2_nonev: np.ndarray,        # (T,) from ModelParams
    mu_alpha: float,                 # prior mean
    sigma2_alpha: float,             # prior variance
) -> tuple[float, float]:
    """Conjugate Gaussian posterior for one home's alpha.

    Model:
      alpha          ~ N(mu_alpha, sigma2_alpha)               [prior]
      beta_hat[t]    ~ N(alpha * rho[t], sigma2_nonev[t]/D)    [likelihood, day-mean obs]

    Returns (posterior_mean, posterior_std).
    With D~350 and T=96, data dominates: posterior_std << prior_std.
    """
    prior_precision = 1.0 / sigma2_alpha
    # Each day-mean observation contributes D times the per-observation precision
    data_precision = num_days * np.sum(rho ** 2 / sigma2_nonev)
    posterior_precision = prior_precision + data_precision

    prior_contribution = mu_alpha / sigma2_alpha
    data_contribution = num_days * np.sum(rho * empirical_day_mean / sigma2_nonev)
    posterior_mean = (prior_contribution + data_contribution) / posterior_precision

    return float(posterior_mean), float(np.sqrt(1.0 / posterior_precision))


def compute_home_background_stats(home_df: pd.DataFrame, params) -> dict:
    """Compute all empirical + model stats for one home's background load.

    Returned dict keys:
      home_id, has_ev, num_days,
      empirical_traces     (D, T)  -- individual daily non-EV traces
      empirical_mean       (T,)    -- mean across days
      empirical_std        (T,)    -- std across days
      posterior_alpha_mean float
      posterior_alpha_std  float
      model_mean           (T,)    -- posterior_alpha_mean * rho
      alpha_only_band_lo/hi (T,)   -- CI from alpha uncertainty alone
      combined_band_lo/hi  (T,)    -- CI from alpha uncertainty + nonev noise
    """
    home_id = int(home_df["home_id"].iloc[0])
    has_ev = bool(home_df["has_ev"].iloc[0])
    num_days = home_df["day"].nunique()

    # Shape individual days into (D, T)
    daily_traces = (
        home_df.groupby(["day", "time_index"])["non_ev_load"]
        .first()
        .unstack("time_index")
        .values
        .astype(np.float64)
    )
    empirical_day_mean = daily_traces.mean(axis=0)          # (T,)
    empirical_day_std  = daily_traces.std(axis=0, ddof=1)   # (T,)

    posterior_alpha_mean, posterior_alpha_std = _compute_alpha_posterior(
        empirical_day_mean = empirical_day_mean,
        num_days           = num_days,
        rho                = params.rho,
        sigma2_nonev       = params.sigma2_nonev,
        mu_alpha           = params.mu_alpha,
        sigma2_alpha       = params.sigma2_alpha,
    )

    model_mean             = posterior_alpha_mean * params.rho                     # (T,)
    alpha_uncertainty_std  = posterior_alpha_std  * params.rho                     # (T,); rho >= 0
    combined_std           = np.sqrt(alpha_uncertainty_std ** 2 + params.sigma2_nonev)

    return dict(
        home_id              = home_id,
        has_ev               = has_ev,
        num_days             = num_days,
        empirical_traces     = daily_traces,
        empirical_mean       = empirical_day_mean,
        empirical_std        = empirical_day_std,
        posterior_alpha_mean = posterior_alpha_mean,
        posterior_alpha_std  = posterior_alpha_std,
        model_mean           = model_mean,
        alpha_only_band_lo   = model_mean - alpha_uncertainty_std,
        alpha_only_band_hi   = model_mean + alpha_uncertainty_std,
        combined_band_lo     = model_mean - combined_std,
        combined_band_hi     = model_mean + combined_std,
    )


def compute_prior_background_stats(params) -> dict:
    """Stats for the prior predictive distribution (no home-specific data)."""
    prior_alpha_std       = np.sqrt(params.sigma2_alpha)
    model_mean            = params.mu_alpha * params.rho
    alpha_uncertainty_std = prior_alpha_std * params.rho
    combined_std          = np.sqrt(alpha_uncertainty_std ** 2 + params.sigma2_nonev)
    return dict(
        model_mean         = model_mean,
        alpha_only_band_lo = model_mean - alpha_uncertainty_std,
        alpha_only_band_hi = model_mean + alpha_uncertainty_std,
        combined_band_lo   = model_mean - combined_std,
        combined_band_hi   = model_mean + combined_std,
        # These fields mirror compute_home_background_stats for uniform drawing code
        empirical_traces   = None,
        empirical_mean     = None,
        empirical_std      = None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-axes drawing (used by both public figure functions)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_background_on_axes(
    ax: Axes,
    stats: dict,
    home_color: str,
    show_individual_traces: bool,
    show_empirical_band: bool,
    show_alpha_band: bool,
    show_combined_band: bool,
    max_trace_days: int,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    is_bottom_row: bool,
) -> None:
    """Draw all requested layers onto a single axes object."""
    time_indices = np.arange(96)

    # ── empirical / true-data layers ─────────────────────────────────────────
    if show_individual_traces and stats["empirical_traces"] is not None:
        traces = stats["empirical_traces"]
        days_to_draw = min(max_trace_days, traces.shape[0])
        sampled_day_indices = np.random.default_rng(0).choice(
            traces.shape[0], days_to_draw, replace=False
        )
        for day_index in sampled_day_indices:
            ax.plot(time_indices, traces[day_index], color=home_color, alpha=0.04, lw=0.5)

    if show_empirical_band and stats["empirical_mean"] is not None:
        ax.plot(
            time_indices, stats["empirical_mean"],
            color=home_color, alpha=0.85, lw=1.8, ls="--",
            label="empirical mean",
        )
        ax.fill_between(
            time_indices,
            stats["empirical_mean"] - stats["empirical_std"],
            stats["empirical_mean"] + stats["empirical_std"],
            color=home_color, alpha=0.18, label="empirical ±1σ",
        )

    # ── model prediction layers ───────────────────────────────────────────────
    # Draw wider (combined) band first so alpha band appears on top
    if show_combined_band:
        ax.fill_between(
            time_indices,
            stats["combined_band_lo"], stats["combined_band_hi"],
            color=home_color, alpha=0.15,
            label=r"model: $\alpha$ + $\sigma^{\mathrm{NonEV}}$ band",
        )

    if show_alpha_band:
        ax.fill_between(
            time_indices,
            stats["alpha_only_band_lo"], stats["alpha_only_band_hi"],
            color=home_color, alpha=0.35,
            label=r"model: $\alpha$-only band",
        )

    # Model mean always shown
    ax.plot(
        time_indices, stats["model_mean"],
        color=home_color, lw=1.5, ls="-", alpha=1.0,
        label="model mean",
    )

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_title(title, fontsize=9, loc="left", pad=3)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.7, ncol=2)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.grid(axis="x", lw=0.2, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    if is_bottom_row:
        _apply_hourly_time_axis(ax)
        ax.set_xlabel("Time of day", fontsize=8)
    else:
        ax.set_xticks([])


# ─────────────────────────────────────────────────────────────────────────────
# Public: stacked time-series comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_background_comparison(
    train_df: pd.DataFrame,
    params,
    home_ids: list[int],
    *,
    show_individual_traces: bool = True,
    show_empirical_band: bool = True,
    show_alpha_band: bool = True,
    show_combined_band: bool = True,
    max_trace_days: int = 60,
    row_height: float = 2.8,
    figure_width: float = 13.0,
    colormap_name: str = "tab20",
) -> Figure:
    """Stacked single-column figure: one row per home + one row for the prior.

    All time-series rows share the same absolute kW y-axis.

    Parameters
    ----------
    home_ids : list of home_id values to include.
    show_individual_traces : draw faint individual daily traces.
    show_empirical_band    : draw empirical mean ± 1 std.
    show_alpha_band        : draw model band from alpha uncertainty alone.
    show_combined_band     : draw model band from alpha + nonev noise.
    max_trace_days         : max individual day traces drawn per home (random sample).
    """
    num_homes = len(home_ids)
    num_rows  = num_homes + 1   # homes + prior

    colormap = plt.get_cmap(colormap_name)
    home_colors = [colormap(i / max(num_homes - 1, 1)) for i in range(num_homes)]

    # ── compute stats ─────────────────────────────────────────────────────────
    all_home_stats = {}
    for home_id in home_ids:
        home_df = train_df[train_df["home_id"] == home_id]
        all_home_stats[home_id] = compute_home_background_stats(home_df, params)

    prior_stats = compute_prior_background_stats(params)

    # ── shared y-axis limits ──────────────────────────────────────────────────
    # Use the empirical band (if available) and model bands to set scale
    all_lower_values = [prior_stats["combined_band_lo"].min()]
    all_upper_values = [prior_stats["combined_band_hi"].max()]
    for stats in all_home_stats.values():
        all_lower_values.append(stats["combined_band_lo"].min())
        all_upper_values.append(stats["combined_band_hi"].max())
        if stats["empirical_mean"] is not None:
            all_lower_values.append((stats["empirical_mean"] - stats["empirical_std"]).min())
            all_upper_values.append((stats["empirical_mean"] + stats["empirical_std"]).max())
    y_min = min(all_lower_values) * 0.9
    y_max = max(all_upper_values) * 1.08
    shared_ylim = (max(0.0, y_min), y_max)   # load is non-negative

    # ── build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        num_rows, 1,
        figsize=(figure_width, row_height * num_rows),
        sharex=False,   # x-axis formatted manually; only bottom row gets labels
    )
    if num_rows == 1:
        axes = [axes]

    # Draw home rows
    for row_index, (home_id, home_color) in enumerate(zip(home_ids, home_colors)):
        stats   = all_home_stats[home_id]
        is_ev   = stats["has_ev"]
        title   = (
            f"Home {home_id}  "
            f"({'EV' if is_ev else 'non-EV'}, {stats['num_days']} days)  "
            f"  α̂ = {stats['posterior_alpha_mean']:.2f} kW"
        )
        _draw_background_on_axes(
            ax                     = axes[row_index],
            stats                  = stats,
            home_color             = home_color,
            show_individual_traces = show_individual_traces,
            show_empirical_band    = show_empirical_band,
            show_alpha_band        = show_alpha_band,
            show_combined_band     = show_combined_band,
            max_trace_days         = max_trace_days,
            title                  = title,
            ylabel                 = "kW",
            ylim                   = shared_ylim,
            is_bottom_row          = False,
        )

    # Draw prior row (last)
    prior_color = "dimgray"
    _draw_background_on_axes(
        ax                     = axes[-1],
        stats                  = prior_stats,
        home_color             = prior_color,
        show_individual_traces = False,   # no individual traces for prior
        show_empirical_band    = False,
        show_alpha_band        = show_alpha_band,
        show_combined_band     = show_combined_band,
        max_trace_days         = 0,
        title                  = (
            f"Prior predictive  "
            f"  μ_α = {params.mu_alpha:.2f} kW,  σ_α = {np.sqrt(params.sigma2_alpha):.2f} kW"
        ),
        ylabel                 = "kW",
        ylim                   = shared_ylim,
        is_bottom_row          = True,
    )

    fig.suptitle("Non-EV background load: empirical vs model", fontsize=11, y=1.002)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: alpha posterior forest plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_posteriors(
    train_df: pd.DataFrame,
    params,
    home_ids: list[int],
    *,
    figure_width: float = 11.0,
    figure_height: float = 5.0,
    colormap_name: str = "tab20",
) -> Figure:
    """Forest plot: posterior alpha ± 1 std for each home, plus the prior.

    x-axis  : x=0 is the global prior; x=1,2,... are individual homes.
    y-axis  : alpha in kW (the per-home background scale factor).

    Note: posterior stds are typically tiny (~0.05 kW) because D*T >> 1,
    so home error bars appear nearly as points compared to the prior CI.
    """
    colormap   = plt.get_cmap(colormap_name)
    num_homes  = len(home_ids)
    home_colors = [colormap(i / max(num_homes - 1, 1)) for i in range(num_homes)]

    # Compute posteriors
    posterior_means = []
    posterior_stds  = []
    for home_id in home_ids:
        home_df = train_df[train_df["home_id"] == home_id]
        stats   = compute_home_background_stats(home_df, params)
        posterior_means.append(stats["posterior_alpha_mean"])
        posterior_stds.append(stats["posterior_alpha_std"])

    prior_mean = params.mu_alpha
    prior_std  = float(np.sqrt(params.sigma2_alpha))

    # x positions: 0 = prior, 1..N = homes
    x_positions  = list(range(num_homes + 1))
    x_prior      = 0
    x_homes      = list(range(1, num_homes + 1))
    x_tick_labels = ["prior"] + [str(h) for h in home_ids]

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # Prior: wide shaded band to make the CI visible at this scale
    ax.axhspan(
        prior_mean - prior_std, prior_mean + prior_std,
        alpha=0.12, color="dimgray", label=f"prior ±1σ  (σ_α={prior_std:.2f} kW)",
    )
    ax.errorbar(
        x_prior, prior_mean, yerr=prior_std,
        fmt="D", color="dimgray", capsize=6, capthick=1.5, elinewidth=1.5, ms=7,
        label=f"prior mean (μ_α={prior_mean:.2f} kW)",
        zorder=5,
    )

    # Home posteriors
    for x_pos, home_id, post_mean, post_std, home_color in zip(
        x_homes, home_ids, posterior_means, posterior_stds, home_colors
    ):
        is_ev = bool(train_df[train_df["home_id"] == home_id]["has_ev"].iloc[0])
        marker = "^" if is_ev else "o"
        ax.errorbar(
            x_pos, post_mean, yerr=post_std,
            fmt=marker, color=home_color,
            capsize=4, capthick=1.2, elinewidth=1.2, ms=7,
            label=f"{home_id}{'*' if is_ev else ''}",
            zorder=4,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("α (kW)", fontsize=9)
    ax.set_xlabel("Home ID  (* = EV home)  |  x=0 is the global prior", fontsize=8)
    ax.set_title(
        "Per-home posterior α ± 1σ  vs  global prior\n"
        "(posterior stds are tiny because D·T ≫ 1; home dots appear near-pointlike)",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8, ncol=3)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.grid(axis="x", lw=0.3, alpha=0.25)

    plt.tight_layout()
    return fig
