"""
Background (Non-EV) visualization for the hierarchical model.

Matches specs/model.md §2.1–§2.5: per-home mean profile eta^(n) under PPCA
prior, per-home std profile omega^(n) under InvGamma prior. All plotting
helpers here are diagnostic / fit-time visualizations, not Gibbs inference.

Public entry points:

  plot_prior_predictive(params)
      Standalone figure of the *generative* prior predictive for a new home:
          center  = eta_bar
          inner   = +/- sqrt(E[omega^2])
          outer   = +/- sqrt(Var(eta) + E[omega^2])

  plot_background_per_home(train_df, params, home_ids)
      Stacked panels (one per home), each comparing:
          empirical day-mean profile +/- empirical day-std band
              (solid line, light fill)
          posterior MAP of eta under fitted prior +/- posterior MAP of omega
              (dashed line, hatched fill)
      Use this to answer: "how finely does the prior allow us to fit this
      home's shape?" — when the prior is tight (small r, small psi), the
      dashed line shrinks toward eta_bar and the dashed band tightens.

  plot_eta_per_home(train_df, params, home_ids)
      Per-home empirical mean profile overlaid on the prior eta distribution
      envelope (eta_bar +/- sqrt(Var(eta))). Answers: "is this home's mean
      typical under the prior?"

  plot_omega_per_home(train_df, params, home_ids)
      Per-home empirical std profile overlaid on the prior omega distribution
      envelope (16/84 IG percentiles). Answers: "is this home's noise
      profile typical under the prior?"

  plot_inference_vs_truth(test_df, inference, params, home_id)
      Inference diagnostic: for one home, compare the inferred posterior over
      (eta, omega) to the ground-truth x_Non-EV statistics. Three stacked
      panels (mean, std, z-error rates). Use this to localize where Gibbs is
      mis-attributing variance — typically, false-positive z timesteps line
      up with timesteps where inferred eta sits below true x_Non-EV mean.

All plots sharing a time axis use hourly HH:MM labels.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats


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
# Computations: prior summaries
# ─────────────────────────────────────────────────────────────────────────────

def _sigma_eta_diag(params) -> np.ndarray:
    """Diagonal of Sigma_eta = W W^T + diag(psi). Shape (T,)."""
    return (params.W_eta ** 2).sum(axis=1) + params.psi_eta


def _omega2_prior_moments(params) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of (omega_t)^2 under the prior on omega.

    For omega_mode='global':
        mean = sigma2_nev_global,  var = 0 (point parameter)
    For omega_mode='hierarchical' (InvGamma(a_omega_t, b_omega_t)):
        mean = b/(a-1), var = b^2 / ((a-1)^2 (a-2))  (NaN if a <= 2)

    Returns (E[omega^2], Var[omega^2]). Both shape (T,).
    """
    if params.omega_mode == "global":
        return params.sigma2_nev_global, np.zeros_like(params.sigma2_nev_global)
    a = params.a_omega
    b = params.b_omega
    mean = b / np.maximum(a - 1.0, 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(
            a > 2.0,
            (b ** 2) / ((a - 1.0) ** 2 * (a - 2.0)),
            np.nan,
        )
    return mean, var


def _omega_prior_quantiles(params, q_lo: float = 0.16, q_hi: float = 0.84) -> tuple[np.ndarray, np.ndarray]:
    """Per-t quantiles of omega_t under the prior.

    For omega_mode='global', prior is a point mass: both quantiles equal
    sqrt(sigma2_nev_global). For omega_mode='hierarchical', uses IG quantiles
    on omega^2 (sqrt for omega).
    """
    if params.omega_mode == "global":
        sig = np.sqrt(params.sigma2_nev_global)
        return sig.copy(), sig.copy()
    omega2_lo = stats.invgamma.ppf(q_lo, a=params.a_omega, scale=params.b_omega)
    omega2_hi = stats.invgamma.ppf(q_hi, a=params.a_omega, scale=params.b_omega)
    return np.sqrt(omega2_lo), np.sqrt(omega2_hi)


def compute_prior_predictive(params) -> dict:
    """Prior predictive summary for a single observation from a NEW home.

    Returns a dict:
        center      : eta_bar                                              (T,)
        sigma_obs   : sqrt(E[omega^2])              (inner band half)      (T,)
        sigma_total : sqrt(Var(eta) + E[omega^2])   (outer band half)      (T,)
        sigma_eta   : sqrt(Var(eta))                                        (T,)
    """
    eta_bar = params.eta_bar
    var_eta = _sigma_eta_diag(params)              # diag of Sigma_eta
    E_omega2, _ = _omega2_prior_moments(params)
    sigma_obs = np.sqrt(np.maximum(E_omega2, 0.0))
    sigma_total = np.sqrt(np.maximum(var_eta + E_omega2, 0.0))
    sigma_eta = np.sqrt(np.maximum(var_eta, 0.0))
    return {
        "center":      eta_bar,
        "sigma_obs":   sigma_obs,
        "sigma_total": sigma_total,
        "sigma_eta":   sigma_eta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Computations: per-home posteriors under the fitted prior
# ─────────────────────────────────────────────────────────────────────────────

def _home_nev_array(home_df: pd.DataFrame, T: int) -> tuple[np.ndarray, int]:
    """Return (x_nev with shape (D, T), D) for one home's rows of train_df."""
    g = home_df.sort_values(["day", "time_index"])
    D = g["day"].nunique()
    if len(g) != D * T:
        raise ValueError(
            f"home has {len(g)} rows, expected D*T = {D*T} (incomplete days?)"
        )
    return g["non_ev_load"].to_numpy().reshape(D, T).astype(np.float64), D


def _sigma_eta_inv(params) -> np.ndarray:
    """Compute Sigma_eta^{-1} via Woodbury identity. (T, T)."""
    W = params.W_eta
    psi = params.psi_eta
    T_ = psi.shape[0]
    r = W.shape[1]
    inv_psi = 1.0 / psi
    if r == 0:
        return np.diag(inv_psi)
    M = np.eye(r) + W.T @ (inv_psi[:, None] * W)
    WP = inv_psi[:, None] * W
    return np.diag(inv_psi) - WP @ np.linalg.solve(M, WP.T)


def compute_home_background_posterior(
    home_df: pd.DataFrame,
    params,
    *,
    Sigma_eta_inv: np.ndarray | None = None,
) -> dict:
    """All quantities needed for the per-home background diagnostic plot.

    The posterior is computed under the fitted prior, conditioning on the
    home's empirical (omega_hat)^2 as the noise scale. This is the same
    plug-in used at Gibbs initialization; with D ≈ 365 obs per t, the
    coupling between eta and omega in the joint posterior is weak.

    Returns:
        empirical_mean : (T,)  hat_eta^(n) = mean over days
        empirical_std  : (T,)  hat_omega^(n) = std over days (ddof=0)
        D              : int   number of days
        posterior_eta_mean : (T,)  MAP/mean of eta posterior (Gaussian)
        posterior_eta_std  : (T,)  sqrt(diag(Lambda^{-1}))
        posterior_omega_map: (T,)  MAP of omega from IG posterior (mode of
                                    sqrt-scale; i.e. sqrt of IG mode)
    """
    T_ = params.T
    x_nev, D = _home_nev_array(home_df, T_)

    emp_mean = x_nev.mean(axis=0)                  # (T,)
    emp_var  = x_nev.var(axis=0, ddof=0)           # (T,)
    emp_std  = np.sqrt(emp_var)

    if Sigma_eta_inv is None:
        Sigma_eta_inv = _sigma_eta_inv(params)

    # Posterior over eta given hat_omega = empirical std
    omega2 = np.maximum(emp_var, 1e-12)            # (T,)
    Lambda = Sigma_eta_inv.copy()
    Lambda.flat[::T_ + 1] += D / omega2            # add D/omega^2 to diagonal
    h = Sigma_eta_inv @ params.eta_bar + D * emp_mean / omega2

    L = np.linalg.cholesky(Lambda)
    post_eta_mean = np.linalg.solve(L.T, np.linalg.solve(L, h))
    # Posterior covariance Lambda^{-1}; diagonal via column-wise solve
    eye_T = np.eye(T_)
    Lambda_inv = np.linalg.solve(L.T, np.linalg.solve(L, eye_T))
    post_eta_std = np.sqrt(np.maximum(np.diag(Lambda_inv), 0.0))

    # Posterior over omega given eta = post_eta_mean.
    # DISPATCH: in 'global' mode omega is fixed (point prior); in
    # 'hierarchical' mode it has an IG posterior whose mode we report.
    if params.omega_mode == "global":
        omega_map = np.sqrt(np.maximum(params.sigma2_nev_global, 0.0))
    else:
        ss_resid = ((x_nev - post_eta_mean[None, :]) ** 2).sum(axis=0)
        a_post = params.a_omega + D / 2.0
        b_post = params.b_omega + 0.5 * ss_resid
        omega2_map = b_post / (a_post + 1.0)
        omega_map = np.sqrt(np.maximum(omega2_map, 0.0))

    return {
        "empirical_mean":     emp_mean,
        "empirical_std":      emp_std,
        "D":                  D,
        "posterior_eta_mean": post_eta_mean,
        "posterior_eta_std":  post_eta_std,
        "posterior_omega_map": omega_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: prior predictive (standalone)
# ─────────────────────────────────────────────────────────────────────────────

def plot_prior_predictive(
    params,
    *,
    figsize: tuple[float, float] = (8.5, 4.0),
) -> Figure:
    """Prior predictive envelope for a new home (one observation, one timestep).

      center      = eta_bar
      inner band  = center +/- sqrt(E[omega^2])                  (avg per-obs noise)
      outer band  = center +/- sqrt(Var(eta) + E[omega^2])       (total predictive)
    """
    pp = compute_prior_predictive(params)
    t = np.arange(params.T)

    fig, ax = plt.subplots(figsize=figsize)

    # Outer band first so it sits behind
    ax.fill_between(
        t,
        pp["center"] - pp["sigma_total"],
        pp["center"] + pp["sigma_total"],
        alpha=0.18, color="C0",
        label=r"$\bar\eta_t \pm \sqrt{\mathrm{Var}(\eta_t) + \mathbb{E}[\omega_t^2]}$  (total)",
    )
    ax.fill_between(
        t,
        pp["center"] - pp["sigma_obs"],
        pp["center"] + pp["sigma_obs"],
        alpha=0.30, color="C0",
        label=r"$\bar\eta_t \pm \sqrt{\mathbb{E}[\omega_t^2]}$  (noise only)",
    )
    ax.plot(t, pp["center"], color="C0", lw=1.8, label=r"$\bar\eta_t$  (prior mean)")

    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Non-EV load (kW)")
    ax.set_title(
        f"Prior predictive for a new home  "
        f"(PPCA rank r={params.ppca_rank}, T={params.T})"
    )
    _apply_hourly_time_axis(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: per-home empirical vs posterior (stacked)
# ─────────────────────────────────────────────────────────────────────────────

def plot_background_per_home(
    train_df: pd.DataFrame,
    params,
    home_ids: Sequence[int],
    *,
    show_individual_traces: bool = False,
    max_trace_days: int = 50,
    panel_height: float = 1.9,
    fig_width: float = 11.0,
) -> Figure:
    """Stacked: per-home empirical envelope vs posterior MAP envelope.

    For each home, one panel:
      solid line  + light fill : empirical mean +/- empirical std (per t)
      dashed line + hatched    : posterior MAP eta +/- posterior MAP omega
                                  (the "model's best guess" under the prior)

    The posterior-MAP envelope is what the prior-constrained model predicts
    a single observation will look like. When the prior is rigid (small r,
    small psi), the dashed line shrinks toward eta_bar; when the prior is
    permissive, dashed ≈ solid.

    All panels share an absolute kW y-axis for cross-home comparability.
    """
    Sigma_eta_inv = _sigma_eta_inv(params)
    home_ids = list(home_ids)
    n_panels = len(home_ids)

    # Pre-compute per-home stats so we can set a shared y-range
    per_home = []
    y_lo, y_hi = +np.inf, -np.inf
    for hid in home_ids:
        g = train_df[train_df["home_id"] == hid]
        if g.empty:
            raise ValueError(f"home_id {hid} not in train_df")
        stats_d = compute_home_background_posterior(g, params, Sigma_eta_inv=Sigma_eta_inv)
        per_home.append((hid, g, stats_d))
        lo = min(
            (stats_d["empirical_mean"] - stats_d["empirical_std"]).min(),
            (stats_d["posterior_eta_mean"] - stats_d["posterior_omega_map"]).min(),
        )
        hi = max(
            (stats_d["empirical_mean"] + stats_d["empirical_std"]).max(),
            (stats_d["posterior_eta_mean"] + stats_d["posterior_omega_map"]).max(),
        )
        y_lo = min(y_lo, lo)
        y_hi = max(y_hi, hi)
    pad = 0.05 * (y_hi - y_lo)
    y_lo -= pad
    y_hi += pad

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(fig_width, panel_height * n_panels),
        sharex=True, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    t = np.arange(params.T)

    for ax, (hid, g, d) in zip(axes, per_home):
        # Optional faint individual day traces
        if show_individual_traces:
            x_nev, D = _home_nev_array(g, params.T)
            n_show = min(D, max_trace_days)
            for row in x_nev[:n_show]:
                ax.plot(t, row, color="0.7", lw=0.4, alpha=0.4)

        # Empirical mean + band
        ax.fill_between(
            t,
            d["empirical_mean"] - d["empirical_std"],
            d["empirical_mean"] + d["empirical_std"],
            alpha=0.22, color="C0",
            label="empirical $\\pm$ day-std" if ax is axes[0] else None,
        )
        ax.plot(
            t, d["empirical_mean"], color="C0", lw=1.5,
            label="empirical mean" if ax is axes[0] else None,
        )

        # Posterior MAP mean + omega-MAP band
        ax.fill_between(
            t,
            d["posterior_eta_mean"] - d["posterior_omega_map"],
            d["posterior_eta_mean"] + d["posterior_omega_map"],
            alpha=0.20, color="C3", hatch="///", edgecolor="C3", linewidth=0,
            label=r"posterior MAP $\eta \pm$ MAP $\omega$" if ax is axes[0] else None,
        )
        ax.plot(
            t, d["posterior_eta_mean"], color="C3", lw=1.4, ls="--",
            label=r"posterior MAP $\eta$" if ax is axes[0] else None,
        )

        has_ev = bool(g["has_ev"].iloc[0])
        ax.set_title(f"home {hid}   D={d['D']}   {'EV' if has_ev else 'non-EV'}",
                     fontsize=9, loc="left")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_ylabel("kW", fontsize=8)
        ax.set_ylim(y_lo, y_hi)

    axes[0].legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    _apply_hourly_time_axis(axes[-1])
    axes[-1].set_xlabel("Time of day")
    fig.suptitle(
        f"Background per-home: empirical (solid) vs posterior-MAP under "
        f"prior (dashed, r={params.ppca_rank})",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: eta distribution comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_eta_per_home(
    train_df: pd.DataFrame,
    params,
    home_ids: Sequence[int],
    *,
    panel_height: float = 1.9,
    fig_width: float = 11.0,
) -> Figure:
    """Per-home empirical mean profile vs prior distribution on eta.

    Shows: does this home's mean profile fall within the prior's typical
    envelope of per-home mean profiles?

      prior center   : eta_bar
      prior band     : eta_bar +/- sqrt(Var(eta_t))   (Sigma_eta diagonal)
      empirical line : hat_eta^(n)_t (mean over days, single home)
    """
    home_ids = list(home_ids)
    n_panels = len(home_ids)
    t = np.arange(params.T)

    eta_bar = params.eta_bar
    sigma_eta = np.sqrt(np.maximum(_sigma_eta_diag(params), 0.0))

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(fig_width, panel_height * n_panels),
        sharex=True, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, hid in zip(axes, home_ids):
        g = train_df[train_df["home_id"] == hid]
        if g.empty:
            raise ValueError(f"home_id {hid} not in train_df")
        x_nev, D = _home_nev_array(g, params.T)
        emp_mean = x_nev.mean(axis=0)

        ax.fill_between(
            t, eta_bar - sigma_eta, eta_bar + sigma_eta,
            alpha=0.25, color="C0",
            label=r"$\bar\eta \pm \sqrt{\mathrm{Var}(\eta_t)}$  (prior)" if ax is axes[0] else None,
        )
        ax.plot(t, eta_bar, color="C0", lw=1.5, ls="--",
                label=r"$\bar\eta$  (prior mean)" if ax is axes[0] else None)
        ax.plot(t, emp_mean, color="C3", lw=1.4,
                label=r"$\hat\eta^{(n)}$  (empirical)" if ax is axes[0] else None)

        has_ev = bool(g["has_ev"].iloc[0])
        ax.set_title(f"home {hid}   D={D}   {'EV' if has_ev else 'non-EV'}",
                     fontsize=9, loc="left")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_ylabel("kW", fontsize=8)

    axes[0].legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    _apply_hourly_time_axis(axes[-1])
    axes[-1].set_xlabel("Time of day")
    fig.suptitle(
        r"Per-home empirical $\hat\eta^{(n)}$ vs prior distribution on $\eta$",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: omega distribution comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_omega_per_home(
    train_df: pd.DataFrame,
    params,
    home_ids: Sequence[int],
    *,
    panel_height: float = 1.9,
    fig_width: float = 11.0,
) -> Figure:
    """Per-home empirical std profile vs prior distribution on omega.

    Shows: does this home's noise profile fall within the prior's typical
    envelope of per-home noise profiles?

      prior center : sqrt(E[omega^2])                     (prior mean of omega)
      prior band   : [16th, 84th] percentiles of omega under IG prior
                      (computed as sqrt of IG quantiles for omega^2)
      empirical    : hat_omega^(n) = empirical std across days
    """
    home_ids = list(home_ids)
    n_panels = len(home_ids)
    t = np.arange(params.T)

    E_omega2, _ = _omega2_prior_moments(params)
    omega_mean = np.sqrt(np.maximum(E_omega2, 0.0))
    omega_lo, omega_hi = _omega_prior_quantiles(params)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(fig_width, panel_height * n_panels),
        sharex=True, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, hid in zip(axes, home_ids):
        g = train_df[train_df["home_id"] == hid]
        if g.empty:
            raise ValueError(f"home_id {hid} not in train_df")
        x_nev, D = _home_nev_array(g, params.T)
        emp_std = np.sqrt(x_nev.var(axis=0, ddof=0))

        ax.fill_between(
            t, omega_lo, omega_hi,
            alpha=0.25, color="C0",
            label=r"$\omega_t$ prior 16/84%" if ax is axes[0] else None,
        )
        ax.plot(t, omega_mean, color="C0", lw=1.5, ls="--",
                label=r"$\sqrt{\mathbb{E}[\omega_t^2]}$  (prior mean)" if ax is axes[0] else None)
        ax.plot(t, emp_std, color="C3", lw=1.4,
                label=r"$\hat\omega^{(n)}$  (empirical)" if ax is axes[0] else None)

        has_ev = bool(g["has_ev"].iloc[0])
        ax.set_title(f"home {hid}   D={D}   {'EV' if has_ev else 'non-EV'}",
                     fontsize=9, loc="left")
        ax.set_ylabel("kW", fontsize=8)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    _apply_hourly_time_axis(axes[-1])
    axes[-1].set_xlabel("Time of day")
    fig.suptitle(
        r"Per-home empirical $\hat\omega^{(n)}$ vs prior distribution on $\omega$",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: inference diagnostic — inferred posterior vs ground-truth x_Non-EV
# ─────────────────────────────────────────────────────────────────────────────

def plot_inference_vs_truth(
    test_df: pd.DataFrame,
    inference,
    params,
    home_id: int,
    *,
    figsize: tuple[float, float] = (11.0, 8.0),
) -> Figure:
    """Compare the inferred (eta, omega) posterior to the ground-truth x_Non-EV.

    Three stacked panels for one home:
      1. Mean profile  : true x_Non-EV day-mean (solid + day-std band) overlaid
                          on inferred eta posterior mean (dashed + posterior std band).
                          Answers: "did the model recover the home's mean shape?"
      2. Std profile   : true x_Non-EV day-std (solid) overlaid on inferred omega
                          posterior mean and 16/84% band.
                          Answers: "did the model recover the home's noise scale?"
      3. z-error rates : per-timestep false-positive (truth=off, pred=charging)
                          and false-negative (truth=charging, pred=off) rates,
                          across days. Useful to see whether mis-classifications
                          line up with where eta inference deviates from truth.

    Requires `non_ev_load` and `charge_state` to be present in test_df (ground
    truth columns).
    """
    g = test_df[test_df["home_id"] == home_id].sort_values(["day", "time_index"])
    if g.empty:
        raise ValueError(f"home_id {home_id} not in test_df")
    T_ = params.T
    D = g["day"].nunique()

    x_nev_true = g["non_ev_load"].to_numpy().reshape(D, T_).astype(np.float64)
    z_true     = g["charge_state"].to_numpy().reshape(D, T_).astype(np.int64)
    has_ev     = bool(g["has_ev"].iloc[0])

    # Truth stats
    emp_mean = x_nev_true.mean(axis=0)
    emp_std  = x_nev_true.std(axis=0, ddof=0)

    # Inferred eta posterior summaries (over retained Gibbs samples)
    if inference.eta_samples is None or inference.omega2_samples is None:
        raise ValueError(
            f"home {home_id}: inference has no eta_samples/omega2_samples — "
            f"was it run with retained samples?"
        )
    eta_mean = inference.eta_samples.mean(axis=0)
    eta_std  = inference.eta_samples.std(axis=0, ddof=0)

    omega_samples = np.sqrt(np.maximum(inference.omega2_samples, 0.0))   # (S, T)
    omega_post_mean = omega_samples.mean(axis=0)
    omega_post_lo, omega_post_hi = np.quantile(omega_samples, [0.16, 0.84], axis=0)

    z_hat = inference.z_hat                                              # (D, T)
    fp_rate = ((z_hat != 0) & (z_true == 0)).mean(axis=0)                # (T,)
    fn_rate = ((z_hat == 0) & (z_true != 0)).mean(axis=0)                # (T,)

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.6, 1.0]},
    )
    t = np.arange(T_)

    # ─── Panel 1: mean ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(
        t, emp_mean - emp_std, emp_mean + emp_std,
        alpha=0.22, color="C0",
        label=r"true $x^{\mathrm{Non\text{-}EV}}$ mean $\pm$ empirical day-std",
    )
    ax.plot(t, emp_mean, color="C0", lw=1.6)
    ax.fill_between(
        t, eta_mean - eta_std, eta_mean + eta_std,
        alpha=0.22, color="C3", hatch="///", edgecolor="C3", linewidth=0,
        label=r"inferred $\eta$ posterior mean $\pm$ posterior std",
    )
    ax.plot(t, eta_mean, color="C3", lw=1.5, ls="--")
    ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("kW (mean profile)")
    ax.set_title(
        f"home {home_id}   D={D}   {'EV' if has_ev else 'non-EV'}   "
        f"C_hat={inference.C_hat}   "
        f"(true C={int(has_ev)})",
        loc="left", fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # ─── Panel 2: std ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, emp_std, color="C0", lw=1.6,
            label=r"true $x^{\mathrm{Non\text{-}EV}}$ empirical day-std")
    ax.fill_between(
        t, omega_post_lo, omega_post_hi,
        alpha=0.22, color="C3",
        label=r"inferred $\omega$ posterior 16/84%",
    )
    ax.plot(t, omega_post_mean, color="C3", lw=1.5, ls="--",
            label=r"inferred $\omega$ posterior mean")
    ax.set_ylabel("kW (std profile)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # ─── Panel 3: z-error rates per timestep ────────────────────────────────
    ax = axes[2]
    ax.bar(t, fp_rate, width=0.9, color="C1", alpha=0.75,
           label="FP rate  (truth=off, pred=charging)")
    ax.bar(t, -fn_rate, width=0.9, color="C2", alpha=0.75,
           label="FN rate  (truth=charging, pred=off)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("fraction of days")
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)

    _apply_hourly_time_axis(axes[-1])
    axes[-1].set_xlabel("Time of day")
    plt.tight_layout()
    return fig
