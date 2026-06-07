"""Standard Gibbs-sampler convergence diagnostics for the C=1 chain.

Complements the trace plots in `infer_plots.plot_convergence` (logL / Θ /
occupancy over iterations) with the usual *quantitative* MCMC diagnostics, run
on the **post-burn-in** scalar chains the C=1 sampler records on a
`HomeInferenceC1` (`theta_trace`, `loglik_trace`, `state_occ_trace`, `S_burn`):

  • Autocorrelation function (ACF)  — how many iterations until draws decorrelate
  • Effective sample size (ESS)     — # of independent draws the chain is worth
  • Running posterior means         — has the estimate stabilised?
  • Split-R̂ (Gelman–Rubin)          — within- vs between-half variance (one chain
                                       split in two; >1.01 ⇒ not yet converged)
  • Geweke z                        — mean of the first 10% vs last 50% of the
                                       chain (|z| ≳ 2 ⇒ drift, not stationary)

Public entry points:
  scalar_chains              -- dict of post-burn-in scalar chains for one home
  autocorr / effective_sample_size / split_rhat / geweke_z   -- numeric helpers
  convergence_table          -- DataFrame of mean/sd/ESS/ESS%/split-R̂/Geweke-z
  print_convergence_summary  -- the table, printed
  plot_autocorrelation       -- ACF small-multiples with white-noise CI band
  plot_running_means         -- running posterior mean per scalar
  plot_convergence_diagnostics -- wrapper: prints the table + both figures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Palette shared with the rest of the notebook (off=purple, low=teal, high=yellow).
_SCALAR_COLORS = {
    "complete-data logL": "#3b528b",
    "Θ_low":  "#21918c",
    "Θ_high": "#fde725",
    "z^EV=off":  "#440154",
    "z^EV=low":  "#21918c",
    "z^EV=high": "#fde725",
}

# Pretty mathtext titles for the plots; the plain keys above are used in the
# printed diagnostic table (where mathtext would just be noise).
_TITLE_MATHTEXT = {
    "Θ_low":      r"$\Theta_{\mathrm{low}}$",
    "Θ_high":     r"$\Theta_{\mathrm{high}}$",
    "z^EV=off":   r"$z^{EV}=\mathrm{off}$",
    "z^EV=low":   r"$z^{EV}=\mathrm{low}$",
    "z^EV=high":  r"$z^{EV}=\mathrm{high}$",
}


def _pretty(label: str) -> str:
    """Mathtext form of a scalar label for plot titles (plain label otherwise)."""
    return _TITLE_MATHTEXT.get(label, label)


# ===========================================================================
# Section 1.  Scalar chains
# ===========================================================================

def scalar_chains(c1, *, include_occupancy: bool = True) -> dict[str, np.ndarray]:
    """Post-burn-in scalar chains for one home's C=1 inference (`HomeInferenceC1`).

    These are the 1-D summaries whose mixing we diagnose: the two free charging
    magnitudes, the complete-data log-likelihood, and (optionally) the per-state
    occupancy fractions.
    """
    sl = slice(c1.S_burn, None)
    # Order chosen so the 3-column grid puts logL / Θ_low / Θ_high on the top
    # row and z^EV=off / low / high beneath them — the "low" and "high" columns
    # then line up vertically.
    chains = {
        "complete-data logL": c1.loglik_trace[sl],
        "Θ_low":  c1.theta_trace[sl, 1],
        "Θ_high": c1.theta_trace[sl, 2],
    }
    if include_occupancy:
        chains["z^EV=off"]  = c1.state_occ_trace[sl, 0]
        chains["z^EV=low"]  = c1.state_occ_trace[sl, 1]
        chains["z^EV=high"] = c1.state_occ_trace[sl, 2]
    return chains


# ===========================================================================
# Section 2.  Numeric diagnostics
# ===========================================================================

def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation ρ_k for k = 0 … max_lag (ρ_0 = 1)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    denom = float(x @ x)
    if denom <= 1e-30:                       # constant chain
        return np.concatenate([[1.0], np.zeros(max_lag)])
    max_lag = min(max_lag, n - 1)
    return np.array([float(x[: n - k] @ x[k:]) / denom for k in range(max_lag + 1)])


def effective_sample_size(x: np.ndarray) -> float:
    """ESS = N / τ, with the integrated autocorrelation time τ estimated by
    Geyer's initial-positive-sequence rule (sum consecutive ACF pairs until a
    pair turns non-positive). Robust to the noisy ACF tail."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return float(n)
    rho = autocorr(x, n - 1)
    if rho[1:].sum() == 0.0:                 # constant / white chain
        return float(n)
    n_pairs = (len(rho) - 1) // 2            # Γ_m = ρ_{2m+1} + ρ_{2m+2}, m ≥ 0
    gamma = rho[1 : 1 + 2 * n_pairs : 2] + rho[2 : 2 + 2 * n_pairs : 2]
    neg = np.where(gamma <= 0)[0]
    cut = neg[0] if neg.size else len(gamma)
    tau = 1.0 + 2.0 * float(gamma[:cut].sum())
    return float(min(n, n / max(tau, 1.0)))


def split_rhat(x: np.ndarray) -> float:
    """Split-R̂ (Gelman–Rubin on one chain split into two halves).

    R̂ → 1 as the chain converges; a common rule of thumb flags R̂ > 1.01."""
    x = np.asarray(x, dtype=np.float64)
    m = len(x) // 2
    if m < 2:
        return float("nan")
    halves = np.array([x[:m], x[len(x) - m:]])      # (2, m)
    W = halves.var(axis=1, ddof=1).mean()           # within-half variance
    if W <= 1e-30:
        return float("nan")
    B = m * halves.mean(axis=1).var(ddof=1)         # between-half variance
    var_hat = (m - 1) / m * W + B / m
    return float(np.sqrt(var_hat / W))


def geweke_z(x: np.ndarray, *, first: float = 0.1, last: float = 0.5) -> float:
    """Geweke z-score: standardised difference between the mean of the first
    `first` and last `last` fractions of the chain, using ESS-adjusted standard
    errors. |z| ≳ 2 suggests the two windows disagree ⇒ not yet stationary."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    a, b = x[: int(first * n)], x[n - int(last * n):]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se2 = a.var(ddof=1) / effective_sample_size(a) + b.var(ddof=1) / effective_sample_size(b)
    if se2 <= 1e-30:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(se2))


def convergence_table(c1, *, include_occupancy: bool = True) -> pd.DataFrame:
    """Per-scalar mean / sd / ESS / ESS% / split-R̂ / Geweke-z over the
    post-burn-in chain."""
    rows = []
    for label, chain in scalar_chains(c1, include_occupancy=include_occupancy).items():
        S = len(chain)
        ess = effective_sample_size(chain)
        rows.append({
            "param":     label,
            "mean":      chain.mean(),
            "sd":        chain.std(),
            "ESS":       ess,
            "ESS/S":     ess / S if S else np.nan,
            "split-R̂":   split_rhat(chain),
            "Geweke z":  geweke_z(chain),
        })
    return pd.DataFrame(rows).set_index("param")


def print_convergence_summary(c1) -> pd.DataFrame:
    """Print (and return) the diagnostic table with a short header."""
    S = len(c1.loglik_trace) - c1.S_burn
    print(f"Gibbs convergence — home {c1.home_id}  "
          f"(burn-in={c1.S_burn}, retained S={S})")
    print("  ESS = effective sample size · split-R̂ → 1 at convergence (flag >1.01) "
          "· |Geweke z| ≳ 2 ⇒ drift")
    tbl = convergence_table(c1)
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(tbl.to_string())
    return tbl


# ===========================================================================
# Section 3.  Plots
# ===========================================================================

def plot_autocorrelation(c1, *, max_lag: int = 50, include_occupancy: bool = True):
    """ACF small-multiples (one panel per scalar) with the white-noise ±95% band.

    Slowly-decaying bars = sticky chain (low ESS); bars dropping inside the band
    by a few lags = good mixing.
    """
    chains = scalar_chains(c1, include_occupancy=include_occupancy)
    n = len(chains)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, (label, chain) in enumerate(chains.items()):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        S = len(chain)
        rho = autocorr(chain, max_lag)
        lags = np.arange(len(rho))
        ci = 1.96 / np.sqrt(S)
        ax.axhspan(-ci, ci, color="0.85", alpha=0.7, label="±95% (white noise)")
        ax.axhline(0, color="k", lw=0.7)
        ax.plot(lags, rho, "-", lw=1.6, color=_SCALAR_COLORS.get(label, "C0"))
        ess = effective_sample_size(chain)
        ax.set_title(f"{_pretty(label)}   (ESS≈{ess:.0f}/{S})", fontsize=9)
        ax.set_xlabel("lag"); ax.set_ylabel("ACF")
        ax.set_ylim(min(-0.15, rho.min() * 1.1), 1.05)
        ax.grid(alpha=0.2)
    fig.suptitle(f"home {c1.home_id} — autocorrelation of retained Gibbs samples",
                 fontsize=11)
    fig.tight_layout()
    return fig


def plot_running_means(c1, *, include_occupancy: bool = True):
    """Running posterior mean per scalar (post-burn-in). A flat tail means the
    Monte-Carlo estimate of that quantity has stabilised."""
    chains = scalar_chains(c1, include_occupancy=include_occupancy)
    n = len(chains)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, (label, chain) in enumerate(chains.items()):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        idx = np.arange(1, len(chain) + 1)
        run = np.cumsum(chain) / idx
        color = _SCALAR_COLORS.get(label, "C0")
        ax.plot(idx, run, color=color, lw=1.6)
        ax.axhline(chain.mean(), color="k", lw=0.8, ls="--", alpha=0.6,
                   label=f"final mean = {chain.mean():.3f}")
        ax.set_title(_pretty(label), fontsize=9)
        ax.set_xlabel("retained sample"); ax.set_ylabel("running mean")
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.2)
    fig.suptitle(f"home {c1.home_id} — running posterior means (post-burn-in)",
                 fontsize=11)
    fig.tight_layout()
    return fig


def plot_convergence_diagnostics(c1, *, max_lag: int = 50,
                                 include_occupancy: bool = True):
    """Print the diagnostic table and draw the ACF + running-mean figures.

    Returns (table, acf_fig, running_fig).
    """
    tbl = print_convergence_summary(c1)
    acf_fig = plot_autocorrelation(c1, max_lag=max_lag,
                                   include_occupancy=include_occupancy)
    run_fig = plot_running_means(c1, include_occupancy=include_occupancy)
    return tbl, acf_fig, run_fig
