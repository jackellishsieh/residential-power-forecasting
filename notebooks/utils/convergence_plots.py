"""
Gibbs sampler convergence diagnostics for the hierarchical Non-EV model.

Public entry points:
  plot_scalar_traces        --  mean(η), Θ_low, Θ_high traces (burn-in + retained)
  plot_eta_trace_heatmap    --  per-t evolution of η across iterations
  plot_loglik_trace         --  complete-data log-likelihood per iteration
  plot_state_occupancy      --  fraction of timesteps in each state per iteration
  plot_running_means        --  cumulative posterior mean of mean(η), Θ (post-burn-in)
  plot_acf                  --  autocorrelation of mean(η), Θ_low, Θ_high
  print_convergence_summary --  ESS and mixing statistics printed to stdout
  plot_all_diagnostics      --  convenience wrapper: all figures + summary

The deprecated rank-1 α scalar has no direct analogue in the hierarchical
model (η is a T-vector). We use mean_t(η[t]) as a coarse scalar summary for
trace / running-mean / ACF panels, and an explicit T-vs-iter heatmap for
the full per-t evolution.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.figure import Figure

STATE_NAMES  = ["off", "low", "high"]
STATE_COLORS = ["dimgray", "cornflowerblue", "tomato"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _effective_sample_size(chain: np.ndarray) -> float:
    """ESS via integrated autocorrelation.  Uses truncated sum up to first negative ACF lag."""
    sample_count = len(chain)
    if sample_count < 4:
        return float(sample_count)
    chain_centered = chain - chain.mean()
    variance       = chain_centered.var()
    if variance < 1e-30:
        return float(sample_count)   # degenerate chain (constant)
    full_acf = np.correlate(chain_centered, chain_centered, mode="full")
    acf      = full_acf[sample_count - 1 :] / (sample_count * variance)
    max_lag = min(sample_count // 2, 500)
    cutoff  = next((lag for lag in range(1, max_lag) if acf[lag] < 0), max_lag)
    integrated_acf_sum = 1.0 + 2.0 * acf[1:cutoff].sum()
    return float(sample_count / max(integrated_acf_sum, 1.0))


def _eta_mean_trace(inference) -> np.ndarray:
    """Per-iteration scalar summary of η: mean over t."""
    return inference.eta_trace.mean(axis=1)


def _post_burnin_chains(inference) -> dict[str, np.ndarray]:
    """Extract named post-burn-in scalar chains from a HomeInference object."""
    S_burn = inference.S_burn
    return {
        "mean(η)": _eta_mean_trace(inference)[S_burn:],
        "Θ_low":   inference.theta_trace[S_burn:, 1],
        "Θ_high":  inference.theta_trace[S_burn:, 2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: trace plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_scalar_traces(inference, *, figure_width: float = 12.0) -> Figure:
    """mean(η), Θ_low, Θ_high traces over all iterations.

    A vertical dashed line marks the end of burn-in.
    The running posterior mean (computed over retained samples only) is
    overlaid so it's easy to see whether it has stabilised.
    """
    eta_mean_trace = _eta_mean_trace(inference)
    n_total = len(eta_mean_trace)
    S_burn  = inference.S_burn
    S       = n_total - S_burn
    iters   = np.arange(1, n_total + 1)

    fig, axes = plt.subplots(3, 1, figsize=(figure_width, 7), sharex=True)

    scalar_info = [
        ("mean(η)", eta_mean_trace,                "steelblue"),
        ("Θ_low",   inference.theta_trace[:, 1],   STATE_COLORS[1]),
        ("Θ_high",  inference.theta_trace[:, 2],   STATE_COLORS[2]),
    ]

    for ax, (label, trace, color) in zip(axes, scalar_info):
        ax.plot(iters, trace, color=color, alpha=0.45, lw=0.7, label="sample")

        retained_trace  = trace[S_burn:]
        running_mean    = np.cumsum(retained_trace) / np.arange(1, S + 1)
        ax.plot(iters[S_burn:], running_mean, color=color, lw=1.8, ls="-",
                label="running mean (post-burn-in)")

        ax.axvline(S_burn, color="k", lw=1.0, ls="--", alpha=0.6, label="burn-in end")

        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Gibbs iteration", fontsize=9)
    fig.suptitle(
        f"Home {inference.home_id}: scalar traces  "
        f"(burn-in={S_burn}, retained={S})",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: η heatmap across iterations
# ─────────────────────────────────────────────────────────────────────────────

def plot_eta_trace_heatmap(inference, *, figure_width: float = 12.0) -> Figure:
    """Per-t η values across all iterations as a heatmap.

    Rows = iterations (top = first iter), columns = time-of-day index.
    Useful for catching pathologies that mean(η) hides: e.g. a particular
    t drifting throughout the chain while the mean stays put.
    """
    eta_trace = inference.eta_trace   # (n_total, T)
    n_total, T_ = eta_trace.shape
    S_burn  = inference.S_burn

    fig, ax = plt.subplots(figsize=(figure_width, 4))
    vmax = np.nanmax(np.abs(eta_trace))
    im = ax.imshow(eta_trace, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.axhline(S_burn - 0.5, color="k", lw=1.0, ls="--", alpha=0.7)
    ax.text(T_ * 1.01, S_burn, "burn-in →", color="k", fontsize=8,
            va="center", ha="left")

    ax.set_xlabel("Time-of-day index t", fontsize=9)
    ax.set_ylabel("Gibbs iteration", fontsize=9)
    ax.set_title(f"Home {inference.home_id}: η trace (per-t evolution)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.025, label="η (kW)")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: log-likelihood trace
# ─────────────────────────────────────────────────────────────────────────────

def plot_loglik_trace(inference, *, figure_width: float = 12.0) -> Figure:
    """Complete-data log-likelihood per iteration.

    Should rise quickly from initialisation and plateau after burn-in.
    A flat or declining log-likelihood post-burn-in signals poor mixing.
    """
    n_total = len(inference.loglik_trace)
    S_burn  = inference.S_burn
    iters   = np.arange(1, n_total + 1)

    fig, ax = plt.subplots(figsize=(figure_width, 3))
    ax.plot(iters, inference.loglik_trace, color="dimgray", lw=0.8, alpha=0.7)
    ax.axvline(S_burn, color="k", lw=1.0, ls="--", alpha=0.6, label="burn-in end")

    window = max(1, n_total // 50)
    smoothed = np.convolve(inference.loglik_trace, np.ones(window) / window, mode="same")
    ax.plot(iters, smoothed, color="crimson", lw=1.5, label=f"rolling mean (w={window})")

    ax.set_xlabel("Gibbs iteration", fontsize=9)
    ax.set_ylabel("log p(x | z, Θ, η, ω)", fontsize=9)
    ax.set_title(f"Home {inference.home_id}: complete-data log-likelihood", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: state occupancy trace
# ─────────────────────────────────────────────────────────────────────────────

def plot_state_occupancy(inference, *, figure_width: float = 12.0) -> Figure:
    """Fraction of (day, time) pairs in each state per iteration."""
    n_total = len(inference.state_occ_trace)
    S_burn  = inference.S_burn
    iters   = np.arange(1, n_total + 1)

    fig, ax = plt.subplots(figsize=(figure_width, 3.5))
    for k, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
        ax.plot(iters, inference.state_occ_trace[:, k],
                color=color, lw=0.8, alpha=0.6, label=name)

    ax.axvline(S_burn, color="k", lw=1.0, ls="--", alpha=0.6, label="burn-in end")
    ax.set_xlabel("Gibbs iteration", fontsize=9)
    ax.set_ylabel("fraction of (d, t) pairs", fontsize=9)
    ax.set_title(f"Home {inference.home_id}: state occupancy per iteration", fontsize=10)
    ax.legend(fontsize=8, loc="right")
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: running means (post-burn-in only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_running_means(inference, *, figure_width: float = 12.0) -> Figure:
    """Cumulative posterior mean of mean(η), Θ_low, Θ_high over retained samples."""
    chains = _post_burnin_chains(inference)
    S      = len(next(iter(chains.values())))
    iters  = np.arange(1, S + 1)

    fig, ax = plt.subplots(figsize=(figure_width, 4))
    colors = ["steelblue", STATE_COLORS[1], STATE_COLORS[2]]

    for (label, chain), color in zip(chains.items(), colors):
        running_mean = np.cumsum(chain) / iters
        ax.plot(iters, running_mean, color=color, lw=1.5, label=label)

    ax.set_xlabel("Retained sample index", fontsize=9)
    ax.set_ylabel("Cumulative posterior mean", fontsize=9)
    ax.set_title(
        f"Home {inference.home_id}: running posterior means  (post-burn-in only)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: autocorrelation
# ─────────────────────────────────────────────────────────────────────────────

def plot_acf(
    inference, *, max_lag: int = 60, figure_width: float = 12.0
) -> Figure:
    """Autocorrelation of mean(η), Θ_low, Θ_high over retained samples."""
    chains = _post_burnin_chains(inference)
    S      = len(next(iter(chains.values())))
    lags   = np.arange(0, min(max_lag + 1, S))
    colors = ["steelblue", STATE_COLORS[1], STATE_COLORS[2]]

    fig, ax = plt.subplots(figsize=(figure_width, 3.5))
    for (label, chain), color in zip(chains.items(), colors):
        centered  = chain - chain.mean()
        variance  = centered.var() + 1e-30
        full_acf  = np.correlate(centered, centered, mode="full")
        acf       = full_acf[S - 1 :] / (S * variance)
        ax.plot(lags, acf[: len(lags)], color=color, lw=1.5, marker=".", ms=3, label=label)

    ax.axhline(0, color="k", lw=0.8)
    ax.axhline( 1.96 / np.sqrt(S), color="k", lw=0.8, ls="--", alpha=0.5, label="±95% CI (white noise)")
    ax.axhline(-1.96 / np.sqrt(S), color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Lag", fontsize=9)
    ax.set_ylabel("Autocorrelation", fontsize=9)
    ax.set_title(f"Home {inference.home_id}: ACF of retained samples", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_convergence_summary(inference) -> None:
    """Print ESS and posterior statistics for scalar chains."""
    chains  = _post_burnin_chains(inference)
    S       = len(next(iter(chains.values())))
    S_burn  = inference.S_burn

    print("─" * 60)
    print(f"Convergence summary  home={inference.home_id}  "
          f"burn-in={S_burn}  retained={S}")
    print("─" * 60)
    print(f"  {'param':<10}  {'mean':>8}  {'std':>8}  {'ESS':>8}  {'ESS/S':>8}")
    for label, chain in chains.items():
        ess   = _effective_sample_size(chain)
        print(f"  {label:<10}  {chain.mean():>8.4f}  {chain.std():>8.4f}  "
              f"{ess:>8.1f}  {ess/S:>8.3f}")

    c_prob = float(inference.c_from_z_samples.mean())
    print(f"\n  P̂(C=1) from z samples  = {c_prob:.4f}"
          f"  →  hard prediction C_hat = {int(c_prob >= 0.5)}")
    print(f"  mean z-transitions/day  = "
          f"{inference.z_transitions_per_day_samples.mean():.2f}")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Public: convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_diagnostics(inference, *, max_acf_lag: int = 60) -> list[Figure]:
    """Plot all diagnostic figures and print the summary.  Returns list of Figure objects."""
    print_convergence_summary(inference)
    return [
        plot_scalar_traces(inference),
        plot_eta_trace_heatmap(inference),
        plot_loglik_trace(inference),
        plot_state_occupancy(inference),
        plot_running_means(inference),
        plot_acf(inference, max_lag=max_acf_lag),
    ]
