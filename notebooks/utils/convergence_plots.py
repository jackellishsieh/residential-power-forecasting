"""
Gibbs sampler convergence diagnostics.

Public entry points:
  plot_scalar_traces        --  α and Θ traces over all iterations (burn-in + retained)
  plot_state_occupancy      --  fraction of timesteps in each state per iteration
  plot_running_means        --  cumulative posterior mean of α, Θ (post-burn-in)
  plot_acf                  --  autocorrelation of α, Θ_low, Θ_high
  print_convergence_summary --  ESS and mixing statistics printed to stdout
  plot_all_diagnostics      --  convenience wrapper: all four figures + summary
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.figure import Figure

# reuse state color convention from ev_plots
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
    # Full autocorrelation via FFT; take positive-lag half
    full_acf = np.correlate(chain_centered, chain_centered, mode="full")
    acf      = full_acf[sample_count - 1 :] / (sample_count * variance)
    # Sum until first negative value (Geyer's initial monotone sequence estimator, simplified)
    max_lag = min(sample_count // 2, 500)
    cutoff  = next((lag for lag in range(1, max_lag) if acf[lag] < 0), max_lag)
    integrated_acf_sum = 1.0 + 2.0 * acf[1:cutoff].sum()
    return float(sample_count / max(integrated_acf_sum, 1.0))


def _post_burnin_chains(inference) -> dict[str, np.ndarray]:
    """Extract named post-burn-in scalar chains from a HomeInference object."""
    S_burn = inference.S_burn
    return {
        "α":      inference.alpha_trace[S_burn:],
        "Θ_low":  inference.theta_trace[S_burn:, 1],
        "Θ_high": inference.theta_trace[S_burn:, 2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: trace plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_scalar_traces(inference, *, figure_width: float = 12.0) -> Figure:
    """α and Θ_low, Θ_high traces over all iterations.

    A vertical dashed line marks the end of burn-in.
    The running posterior mean (computed over retained samples only) is
    overlaid so it's easy to see whether it has stabilised.
    """
    n_total = len(inference.alpha_trace)
    S_burn  = inference.S_burn
    S       = n_total - S_burn
    iters   = np.arange(1, n_total + 1)

    fig, axes = plt.subplots(3, 1, figsize=(figure_width, 7), sharex=True)

    scalar_info = [
        ("α",      inference.alpha_trace,         "steelblue"),
        ("Θ_low",  inference.theta_trace[:, 1],   STATE_COLORS[1]),
        ("Θ_high", inference.theta_trace[:, 2],   STATE_COLORS[2]),
    ]

    for ax, (label, trace, color) in zip(axes, scalar_info):
        # Full trace (faint)
        ax.plot(iters, trace, color=color, alpha=0.45, lw=0.7, label="sample")

        # Running mean over retained samples only
        retained_trace  = trace[S_burn:]
        running_mean    = np.cumsum(retained_trace) / np.arange(1, S + 1)
        ax.plot(iters[S_burn:], running_mean, color=color, lw=1.8, ls="-",
                label="running mean (post-burn-in)")

        # Burn-in boundary
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

    # Smoothed rolling mean to show trend more clearly
    window = max(1, n_total // 50)
    smoothed = np.convolve(inference.loglik_trace, np.ones(window) / window, mode="same")
    ax.plot(iters, smoothed, color="crimson", lw=1.5, label=f"rolling mean (w={window})")

    ax.set_xlabel("Gibbs iteration", fontsize=9)
    ax.set_ylabel("log p(x | z, θ, α)", fontsize=9)
    ax.set_title(f"Home {inference.home_id}: complete-data log-likelihood", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: state occupancy trace
# ─────────────────────────────────────────────────────────────────────────────

def plot_state_occupancy(inference, *, figure_width: float = 12.0) -> Figure:
    """Fraction of (day, time) pairs in each state per iteration.

    Since FFBS is exact, this series mixes quickly — it mainly tells you
    whether the overall charging load balance is stable post-burn-in.
    """
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
    """Cumulative posterior mean of α, Θ_low, Θ_high over retained samples.

    A flat line indicates the estimate has converged; still-drifting lines
    suggest more samples are needed.
    """
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
    """Autocorrelation of α, Θ_low, Θ_high over retained samples.

    Fast decay (within ~10 lags) indicates good mixing.
    Slow decay suggests the chain is exploring slowly — consider thinning
    or a longer burn-in, though for this model it's rarely needed.
    """
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
    colors  = {"α": "steelblue", "Θ_low": "cornflowerblue", "Θ_high": "tomato"}
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

    # C^(n) estimate
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
    """Plot all four diagnostic figures and print the summary.  Returns list of Figure objects."""
    print_convergence_summary(inference)
    return [
        plot_scalar_traces(inference),
        plot_loglik_trace(inference),
        plot_state_occupancy(inference),
        plot_running_means(inference),
        plot_acf(inference, max_lag=max_acf_lag),
    ]
