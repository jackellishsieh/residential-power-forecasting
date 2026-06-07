"""Per-home inference under the two EV hypotheses, and the C=0 vs C=1 comparison.

The two hypotheses share NO information, so we infer them with two completely
separate procedures (specs/model.md §4):

    C = 0   (no EV):  z^EV is pinned to off, Theta is irrelevant, and the model
                      collapses to the Non-EV daily LDS.  The posterior over
                      z^LDS is then *exact* — one Kalman smoother call — and the
                      marginal log p(x | C=0) is the Kalman marginal likelihood.
                      → infer_home_c0  (no sampling)

    C = 1   (EV):     three-block Gibbs over (Theta, z^EV, z^LDS):
                        Block A.  z^EV | Theta, z^LDS   — HMM forward-filter
                                  backward-sample (C=1 fixed, so z^EV is free).
                        Block B.  Theta_k | z^EV, z^LDS — truncated-Normal conj.
                        Block C.  z^LDS | z^EV, Theta   — Kalman FFBS on residuals.
                      → infer_home_c1  (Gibbs)

Model comparison.  We want log p(x, C=c) = log p(C=c) + log p(x | C=c):

    log p(x, C=0)  is closed-form (Kalman marginal, z^LDS integrated out).

    log p(x, C=1)  is intractable (marginalising the (D×T) discrete z^EV).
                   We report two estimates (see specs/model.md §5):
      (A)  plug-in joint complete-data density at a representative posterior
           point (z^EV*, z^LDS*, Theta*).  Cheap but biased — it conditions on a
           single z^LDS instead of integrating it, which penalises C=1 by the
           z^LDS dimensionality.  `log_evidence_c1_plugin`.
      (A') the same but with z^LDS integrated out exactly via the Kalman marginal
           (only the discrete z^EV is plugged in).  Far less biased and nearly
           free; it is also the first two terms of the unbiased Chib estimator
           (B, not yet implemented).  `log_evidence_c1_rb_zev`.

The per-home Gibbs retains the z^EV MAP and (optionally) the z^EV samples, so the
Chib correction `- log p(z^EV* | x, C=1, Theta*)` can be added later to turn (A')
into the exact marginal.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import truncnorm

from . import ev, non_ev_lds
from .params import (
    ChibResult, HomeInferenceC0, HomeInferenceC1, HomeResult, K, ModelParams,
    STATE_NAMES, T, THETA_BOUNDS, THETA_VAR_FLOOR,
)


# ===========================================================================
# Section 1.  Emission log-likelihoods conditional on the latents
# ===========================================================================

def compute_loglik(
    x:          np.ndarray,     # (D, T)
    z:          np.ndarray,     # (D, T)
    theta:      np.ndarray,     # (K,)
    nonev_mean: np.ndarray,     # (D, T)   per-cell Non-EV offset, e.g. (C z^LDS)[d, t]
    nonev_var:  np.ndarray,     # (T,)     per-t Non-EV emission variance, e.g. diag(R)[t]
    params: ModelParams,
) -> float:
    """Complete-data emission log-density given z^EV and the Non-EV offset:

        Σ_{d,t} log N( x[d,t] ; theta[z[d,t]] + nonev_mean[d,t],
                                sigma2_ev[z[d,t]] + nonev_var[t] )

    This conditions on a *specific* z^LDS (via nonev_mean); it does NOT integrate
    z^LDS out.  Use `lds_loglik_c1_given_zev` for the z^LDS-marginal version.
    """
    var_dt  = params.sigma2_ev[z] + nonev_var[None, :]
    mean_dt = theta[z] + nonev_mean
    ll = -0.5 * (np.log(2 * np.pi * var_dt) + (x - mean_dt) ** 2 / var_dt)
    return float(ll.sum())


def compute_loglik_c0(
    x:          np.ndarray,     # (D, T)
    nonev_mean: np.ndarray,     # (D, T)
    nonev_var:  np.ndarray,     # (T,)
    params: ModelParams,
) -> float:
    """log p(x | C=0, z^LDS, Theta) for a *specific* z^LDS (z^EV ≡ off):

        Σ_{d,t} log N( x[d,t] ; nonev_mean[d,t],  sigma2_ev[off] + nonev_var[t] )
    """
    var_t    = params.sigma2_ev[0] + nonev_var          # (T,)
    residual = x - nonev_mean                           # (D, T)
    ll = -0.5 * (np.log(2 * np.pi * var_t[None, :])
                 + residual ** 2 / var_t[None, :])
    return float(ll.sum())


# ===========================================================================
# Section 2.  Marginal likelihoods (z^LDS integrated out via Kalman) + priors
# ===========================================================================

def lds_loglik_c0(home_x: np.ndarray, params: ModelParams) -> float:
    """Exact log p(x | C=0) — z^EV pinned off, z^LDS integrated out by the Kalman
    filter.  The off-state EV emission variance sigma2_ev[off] is added to diag(R)
    so this is consistent with the C=1 expression at z^EV ≡ off.
    """
    D = home_x.shape[0]
    extra = np.full((D, T), params.sigma2_ev[0], dtype=np.float64)   # (D, T)
    return float(params.lds.loglik(home_x, extra_obs_cov=extra))


def lds_loglik_c1_given_zev(
    home_x: np.ndarray,         # (D, T)
    z_ev:   np.ndarray,         # (D, T)  int
    theta:  np.ndarray,         # (K,)
    params: ModelParams,
) -> float:
    """log p(x | C=1, z^EV, Theta), with z^LDS integrated out exactly.

    Given z^EV and Theta, the residual x - Theta[z^EV] is a linear-Gaussian
    observation of the LDS with heteroscedastic per-cell noise sigma2_ev[z^EV];
    the Kalman marginal likelihood integrates z^LDS analytically.
    """
    residual = home_x - theta[z_ev]                     # (D, T)
    extra    = params.sigma2_ev[z_ev]                   # (D, T)
    return float(params.lds.loglik(residual, extra_obs_cov=extra))


def log_hmm_prior(z_ev: np.ndarray, params: ModelParams) -> float:
    """log p(z^EV | C=1) under the per-day HMM (days are independent chains):

        Σ_d [ log pi[z_{d,0}] + Σ_{t≥1} log P[z_{d,t-1}, z_{d,t}] ]
    """
    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)
    init   = log_pi[z_ev[:, 0]].sum()
    trans  = log_P[z_ev[:, :-1], z_ev[:, 1:]].sum()
    return float(init + trans)


def log_lds_prior(z_lds: np.ndarray, params: ModelParams) -> float:
    """log p(z^LDS_{1:D}) under the LDS Gaussian-chain prior:

        log N(z_1; mu_0, Sigma_0) + Σ_{d≥2} log N(z_d; A z_{d-1}, Q)

    z_lds is (D, L).  Sigma_0 and Q are diagonal in the current setup but this is
    written for the general PSD case via Cholesky.
    """
    lds  = params.lds
    D, L = z_lds.shape

    # d = 1 term, cov Sigma_0
    r0 = (z_lds[0] - lds.mu_0)[None, :]                 # (1, L)
    ll = _gauss_logpdf_chol(r0, np.linalg.cholesky(lds.Sigma_0))

    # d = 2..D terms, cov Q, mean A z_{d-1}
    if D >= 2:
        pred = z_lds[:-1] @ lds.A.T                     # (D-1, L)
        rd   = z_lds[1:] - pred                         # (D-1, L)
        ll  += _gauss_logpdf_chol(rd, np.linalg.cholesky(lds.Q))
    return float(ll)


def _gauss_logpdf_chol(residuals: np.ndarray, chol: np.ndarray) -> float:
    """Σ_n log N(residuals[n]; 0, Σ) summed over rows, given chol with Σ = chol cholᵀ."""
    L = residuals.shape[1]
    y = np.linalg.solve(chol, residuals.T)              # (L, N)
    quad = np.einsum("ln,ln->n", y, y)                  # (N,)
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    return float((-0.5 * (L * np.log(2 * np.pi) + logdet + quad)).sum())


def log_joint_c1_plugin(
    home_x: np.ndarray,         # (D, T)
    z_ev:   np.ndarray,         # (D, T)  int
    z_lds:  np.ndarray,         # (D, L)
    theta:  np.ndarray,         # (K,)
    params: ModelParams,
) -> float:
    """Plug-in complete-data joint log p(x, z^EV, z^LDS | C=1, Theta) at one point:

        log p(z^EV | HMM) + log p(z^LDS | LDS) + log p(x | z^EV, z^LDS, Theta)

    This is option (A): it conditions on the given z^LDS rather than integrating
    it, so it under-rates C=1 relative to the true marginal.
    """
    nonev_mean = z_lds @ params.lds.C.T                 # (D, T)
    nonev_var  = params.lds.diag_R()                    # (T,)
    return (log_hmm_prior(z_ev, params)
            + log_lds_prior(z_lds, params)
            + compute_loglik(home_x, z_ev, theta, nonev_mean, nonev_var, params))


def log_theta_prior(theta: np.ndarray, params: ModelParams) -> float:
    """log p(Theta) under the truncated-Normal magnitude prior (§1.5):

        Σ_{k∈{low,high}} log TruncNormal(theta[k]; mu_theta[k], sigma_theta[k], [lb,ub])

    Theta_off is pinned to 0 and contributes nothing.
    """
    lp = 0.0
    for k in (1, 2):
        mu     = float(params.mu_theta[k])
        sd     = float(np.sqrt(max(params.sigma2_theta[k], THETA_VAR_FLOOR)))
        lb, ub = THETA_BOUNDS[k]
        lp += float(truncnorm.logpdf(theta[k], (lb - mu) / sd, (ub - mu) / sd,
                                     loc=mu, scale=sd))
    return lp


def _log_theta_cond(
    theta_star: np.ndarray,     # (K,)
    home_x:     np.ndarray,     # (D, T)
    z_ev:       np.ndarray,     # (D, T)  int
    nonev_mean: np.ndarray,     # (D, T)
    nonev_var:  np.ndarray,     # (T,)
    params:     ModelParams,
) -> float:
    """log p(Theta* | x, z^EV, z^LDS) — the Gibbs full conditional density at Theta*.

    The two states k ∈ {low, high} are conditionally independent given (z^EV,
    z^LDS), so the joint density is the sum of the per-k truncated-Normal log
    densities at their conditional posterior params.
    """
    lp = 0.0
    for k in (1, 2):
        m, sd, lb, ub = ev.theta_k_posterior_params(home_x, z_ev, nonev_mean,
                                                     nonev_var, params, k)
        lp += float(truncnorm.logpdf(theta_star[k], (lb - m) / sd, (ub - m) / sd,
                                     loc=m, scale=sd))
    return lp


def _log_p_zev_given(
    home_x:     np.ndarray,     # (D, T)
    z_ev:       np.ndarray,     # (D, T)  int — the path to score
    theta:      np.ndarray,     # (K,)
    nonev_mean: np.ndarray,     # (D, T)
    nonev_var:  np.ndarray,     # (T,)
    params:     ModelParams,
    log_pi:     np.ndarray,
    log_P:      np.ndarray,
) -> float:
    """log p(z^EV | x, Theta, z^LDS) — HMM full-conditional path probability.

        p(z^EV | x, Θ, z^LDS) = p(z^EV, x | Θ, z^LDS) / p(x | Θ, z^LDS)

    Numerator (joint of the path and the obs) = HMM prior of the path + Σ log
    emissions at the path; denominator = exp(log_Z1), the HMM forward marginal.
    """
    log_emit = compute_loglik(home_x, z_ev, theta, nonev_mean, nonev_var, params)
    _, log_Z1 = ev.hmm_forward(home_x, theta, nonev_mean, nonev_var,
                               params, log_pi, log_P)
    return log_hmm_prior(z_ev, params) + log_emit - log_Z1


def _logmeanexp(values: np.ndarray) -> float:
    """log( (1/N) Σ exp(values) ) — numerically stable Monte-Carlo average in log-space."""
    a = np.asarray(values, dtype=np.float64)
    return float(logsumexp(a) - np.log(len(a)))


# ===========================================================================
# Section 3.  C=0 inference — single exact smoother pass
# ===========================================================================

def infer_home_c0(
    home_x: np.ndarray,
    params: ModelParams,
    *,
    home_id: int = -1,
    verbose: bool = True,
) -> HomeInferenceC0:
    """Exact C=0 inference for one home — specs/model.md §4.1.

    Under C=0 the only latent is z^LDS, whose posterior is Gaussian and exact:
    one RTS smoother call gives its mean/cov, and the Kalman filter gives the
    marginal log p(x | C=0).  No sampling.
    """
    D = home_x.shape[0]
    extra = np.full((D, T), params.sigma2_ev[0], dtype=np.float64)

    sm = params.lds.smooth(home_x, extra_obs_cov=extra)
    C_lds = params.lds.C

    # Posterior mean/var in observation space (C z^LDS_d).
    z_lds_mean = sm.z_smooth @ C_lds.T                                  # (D, T)
    # diag of C P_smooth Cᵀ per day → per-cell predictive variance of the Non-EV mean
    z_lds_cov_diag = np.einsum("tl,dlm,tm->dt", C_lds, sm.P_smooth, C_lds)

    log_lik      = float(sm.log_lik)
    log_evidence = float(np.log(1.0 - params.p_C + 1e-300)) + log_lik

    if verbose:
        print(f"  [home {home_id}] C=0 (exact): "
              f"log p(x|C=0)={log_lik:+.1f}  log p(x,C=0)={log_evidence:+.1f}")

    return HomeInferenceC0(
        home_id        = home_id,
        z_lds_mean     = z_lds_mean,
        z_lds_cov_diag = z_lds_cov_diag,
        log_lik        = log_lik,
        log_evidence   = log_evidence,
    )


# ===========================================================================
# Section 4.  C=1 inference — three-block Gibbs (Theta, z^EV, z^LDS)
# ===========================================================================

def infer_home_c1(
    home_x: np.ndarray,
    params: ModelParams,
    *,
    S_burn: int = 200,
    S: int = 500,
    rng: np.random.Generator | None = None,
    home_id: int = -1,
    verbose: bool = True,
    record_traces: bool = True,
    retain_z_ev: bool = False,
    compute_chib: bool = False,
    M_chib: int = 100,
    G_chib_burn: int = 100,
    G_chib: int = 300,
) -> HomeInferenceC1:
    """Three-block Gibbs under C=1 fixed — specs/model.md §4.2.

    home_x : (D, T) total grid power — the only signal at test time.

    Blocks per sweep:
        A.  z^EV | Theta, z^LDS   — HMM forward-filter backward-sample (C=1).
        B.  Theta_k | z^EV, z^LDS — truncated-Normal conjugate, k ∈ {low, high}.
        C.  z^LDS | z^EV, Theta   — Kalman FFBS on residuals x - Theta[z^EV].

    Set `retain_z_ev=True` to store every retained z^EV sample ((S, D, T) int8).

    Set `compute_chib=True` to additionally compute the exact marginal
    log p(x | C=1) via the Chib estimator (B, specs §5): up to `M_chib`
    (Theta, z^LDS) main-run draws are retained for the z^EV posterior ordinate,
    and a `G_chib_burn + G_chib` reduced run (z^EV fixed at the MAP) estimates
    the Theta ordinate. Result lands in `log_evidence_chib` / `chib`.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"

    if verbose:
        print(f"  [home {home_id}] D={D} → "
              f"C=1 Gibbs ({S_burn} burn-in + {S} retained)")

    lds       = params.lds
    C_lds     = lds.C                                   # (T, L)
    nonev_var = lds.diag_R()                            # (T,)
    log_pi    = np.log(params.pi_z + 1e-300)
    log_P     = np.log(params.P_z  + 1e-300)

    # ── initialise ──────────────────────────────────────────────────────────
    theta = params.mu_theta.copy()
    z_lds = lds.smooth(home_x).z_smooth                 # warm start: treat x as Non-EV
    z     = np.zeros((D, T), dtype=np.int64)

    # ── storage ─────────────────────────────────────────────────────────────
    n_total       = S_burn + S
    z_counts      = np.zeros((D, T, K), dtype=np.float64)
    theta_samples = np.zeros((S, K),    dtype=np.float64)
    z_lds_mean    = np.zeros((D, T),    dtype=np.float64)
    z_ev_samples  = (np.zeros((S, D, T), dtype=np.int8) if retain_z_ev else None)

    # Chib: keep a strided subsample of (Theta, z^LDS) main-run draws from the
    # full posterior, used to estimate the z^EV posterior ordinate (specs §5, B).
    chib_draws: list[tuple[np.ndarray, np.ndarray]] = []
    chib_stride = max(1, S // M_chib) if compute_chib else 0

    if record_traces:
        theta_trace     = np.zeros((n_total, K), dtype=np.float64)
        state_occ_trace = np.zeros((n_total, K), dtype=np.float64)
        loglik_trace    = np.zeros(n_total,      dtype=np.float64)
    else:
        theta_trace = state_occ_trace = loglik_trace = None

    # ── main loop ───────────────────────────────────────────────────────────
    t_start = time.time()
    z_lds_last: np.ndarray | None = None
    z_ev_last:  np.ndarray | None = None

    for it in range(n_total):
        nonev_mean = z_lds @ C_lds.T                                    # (D, T)

        # Block A: z^EV | Theta, z^LDS  (C=1 fixed → free HMM FFBS).
        z, _ = ev.ffbs(home_x, theta, nonev_mean, nonev_var,
                       params, log_pi, log_P, rng)

        # Block B: Theta_k | z^EV, z^LDS.
        for k in (1, 2):
            theta[k] = ev.sample_theta_k(home_x, z, nonev_mean, nonev_var,
                                          params, k, rng)

        # Block C: z^LDS | z^EV, Theta.
        z_lds = non_ev_lds.sample_z_lds(home_x, z, theta, params.sigma2_ev,
                                         lds, rng)                      # (D, L)

        # ── record ────────────────────────────────────────────────────────
        if record_traces:
            theta_trace[it]     = theta
            state_occ_trace[it] = [(z == k).mean() for k in range(K)]
            loglik_trace[it]    = compute_loglik(home_x, z, theta, nonev_mean,
                                                  nonev_var, params)

        if it >= S_burn:
            s_idx                = it - S_burn
            theta_samples[s_idx] = theta
            for k in range(K):
                z_counts[:, :, k] += (z == k)
            z_lds_mean += (z_lds @ C_lds.T - z_lds_mean) / (s_idx + 1)
            z_lds_last  = (z_lds @ C_lds.T).copy()
            z_ev_last   = z.copy()
            if z_ev_samples is not None:
                z_ev_samples[s_idx] = z.astype(np.int8)
            if compute_chib and s_idx % chib_stride == 0 and len(chib_draws) < M_chib:
                chib_draws.append((theta.copy(), z_lds.copy()))

        if verbose and (it < 3 or it == S_burn or (it + 1) % 100 == 0):
            phase   = "burn-in" if it < S_burn else "keep  "
            elapsed = time.time() - t_start
            ll = loglik_trace[it] if record_traces else float("nan")
            print(f"    iter {it+1:4d}/{n_total} [{phase}]  "
                  f"Θ_low={theta[1]:.3f}  Θ_high={theta[2]:.3f}  "
                  f"logL={ll:.1f}  ({elapsed:.1f}s)")

    # ── summaries ───────────────────────────────────────────────────────────
    z_marginals = z_counts / S
    z_hat       = np.argmax(z_marginals, axis=2)
    theta_mean  = theta_samples.mean(axis=0)

    # Representative posterior point for the plug-in evidence (option A):
    # z^EV* = MAP marginal, Theta* = posterior mean, z^LDS* = its conditional
    # smoother mean (one Kalman smoother on the residual under z^EV*, Theta*).
    resid_star    = home_x - theta_mean[z_hat]
    extra_star    = params.sigma2_ev[z_hat]
    z_lds_star    = lds.smooth(resid_star, extra_obs_cov=extra_star).z_smooth

    log_joint_plugin   = log_joint_c1_plugin(home_x, z_hat, z_lds_star,
                                              theta_mean, params)
    log_prior_c1       = float(np.log(params.p_C + 1e-300))
    log_evidence_plug  = log_prior_c1 + log_joint_plugin
    # A': z^LDS integrated out exactly, only z^EV plugged in (bridge to Chib B).
    log_evidence_rb    = (log_prior_c1
                          + log_hmm_prior(z_hat, params)
                          + lds_loglik_c1_given_zev(home_x, z_hat, theta_mean, params))

    if verbose:
        elapsed = time.time() - t_start
        frac = z_marginals.mean(axis=(0, 1))
        print(f"  [home {home_id}] C=1 done in {elapsed:.1f}s")
        print(f"    z freq : off={frac[0]:.3f}  low={frac[1]:.3f}  high={frac[2]:.3f}")
        for k in (1, 2):
            print(f"    Θ[{STATE_NAMES[k]:>4}] : "
                  f"mean={theta_samples[:,k].mean():.3f}  "
                  f"std={theta_samples[:,k].std():.4f}")
        print(f"    log p(x,C=1)  plug-in={log_evidence_plug:+.1f}  "
              f"(A′ z^LDS-marg={log_evidence_rb:+.1f})")

    # ── exact marginal via Chib (B), if requested ───────────────────────────
    chib_res = None
    log_evidence_chib = None
    if compute_chib:
        chib_res = chib_marginal_loglik_c1(
            home_x, params,
            z_ev_star=z_hat, theta_star=theta_mean, main_draws=chib_draws,
            G_burn=G_chib_burn, G=G_chib, rng=rng, verbose=verbose,
        )
        log_evidence_chib = chib_res.log_evidence

    return HomeInferenceC1(
        home_id             = home_id,
        z_hat               = z_hat,
        z_marginals         = z_marginals,
        theta_samples       = theta_samples,
        theta_mean          = theta_mean,
        z_lds_mean          = z_lds_mean,
        z_lds_last          = z_lds_last,
        z_ev_last           = z_ev_last,
        z_lds_star          = z_lds_star @ C_lds.T,
        log_joint_plugin    = float(log_joint_plugin),
        log_evidence_plugin = float(log_evidence_plug),
        log_evidence_rb     = float(log_evidence_rb),
        log_evidence_chib   = log_evidence_chib,
        chib                = chib_res,
        theta_trace         = theta_trace,
        state_occ_trace     = state_occ_trace,
        loglik_trace        = loglik_trace,
        z_ev_samples        = z_ev_samples,
        S_burn              = S_burn,
    )


# ===========================================================================
# Section 4b.  Chib (1995) marginal likelihood for C=1  (specs §5, estimator B)
# ===========================================================================

def chib_marginal_loglik_c1(
    home_x: np.ndarray,
    params: ModelParams,
    *,
    z_ev_star:  np.ndarray,                 # (D, T) int — fixed high-density EV path
    theta_star: np.ndarray,                 # (K,)       — fixed high-density magnitudes
    main_draws: list,                       # [(theta (K,), z_lds (D,L))] from p(Θ,z^LDS | x, C=1)
    G_burn: int = 100,
    G: int = 300,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> ChibResult:
    """Exact log p(x | C=1) via Chib (1995) at the point ψ* = (z^EV*, Θ*, z^LDS*).

        log p(x|C=1) = log p(x|ψ*) + log p(ψ*) − log p(ψ*|x)

    z^LDS* is taken as the conditional smoother mean at (z^EV*, Θ*). The three
    posterior-ordinate terms (Gibbs block order z^EV → Θ → z^LDS):

      • log p(z^EV* | x)            — average of the HMM path probability over the
                                      supplied full-posterior `main_draws`.
      • log p(Θ* | x, z^EV*)        — average of the Θ full-conditional density over
                                      a reduced run (z^EV fixed at z^EV*, sampling
                                      Θ and z^LDS).
      • log p(z^LDS* | x, z^EV*, Θ*)— closed-form FFBS Gaussian density.

    The Gaussian sub-identity `log p(x|ψ*) + log p(z^LDS*) − log p(z^LDS*|x,…)`
    must reproduce `lds_loglik_c1_given_zev`; both are returned for a cross-check.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not main_draws:
        raise ValueError("chib_marginal_loglik_c1 needs main_draws "
                         "(run infer_home_c1 with compute_chib=True)")

    lds       = params.lds
    C_lds     = lds.C
    nonev_var = lds.diag_R()
    log_pi    = np.log(params.pi_z + 1e-300)
    log_P     = np.log(params.P_z  + 1e-300)

    # z^LDS* = conditional smoother mean at (z^EV*, Θ*).
    resid_star  = home_x - theta_star[z_ev_star]
    extra_star  = params.sigma2_ev[z_ev_star]
    z_lds_star  = lds.smooth(resid_star, extra_obs_cov=extra_star).z_smooth      # (D, L)
    nonev_star  = z_lds_star @ C_lds.T

    # Term 1 — log p(z^EV* | x): average HMM path prob over full-posterior draws.
    zev_vals = np.array([
        _log_p_zev_given(home_x, z_ev_star, th_m, zl_m @ C_lds.T,
                         nonev_var, params, log_pi, log_P)
        for (th_m, zl_m) in main_draws
    ])
    ord_zev = _logmeanexp(zev_vals)

    # Term 2 — log p(Θ* | x, z^EV*): reduced run with z^EV fixed at z^EV*.
    theta_g = theta_star.copy()
    z_lds_g = z_lds_star.copy()
    theta_cond_vals = []
    for g in range(G_burn + G):
        nonev_g = z_lds_g @ C_lds.T
        for k in (1, 2):
            theta_g[k] = ev.sample_theta_k(home_x, z_ev_star, nonev_g,
                                            nonev_var, params, k, rng)
        z_lds_g = non_ev_lds.sample_z_lds(home_x, z_ev_star, theta_g,
                                          params.sigma2_ev, lds, rng)
        if g >= G_burn:
            theta_cond_vals.append(
                _log_theta_cond(theta_star, home_x, z_ev_star,
                                z_lds_g @ C_lds.T, nonev_var, params))
    ord_theta = _logmeanexp(np.array(theta_cond_vals))

    # Term 3 — log p(z^LDS* | x, z^EV*, Θ*): closed-form FFBS density.
    ord_zlds = non_ev_lds.kalman_logpdf(lds, resid_star, z_lds_star,
                                        extra_obs_cov=extra_star)

    # Assemble the identity.
    log_lik_star   = compute_loglik(home_x, z_ev_star, theta_star, nonev_star,
                                    nonev_var, params)
    log_prior_star = (log_hmm_prior(z_ev_star, params)
                      + log_theta_prior(theta_star, params)
                      + log_lds_prior(z_lds_star, params))
    log_post_star  = ord_zev + ord_theta + ord_zlds
    log_lik_c1     = log_lik_star + log_prior_star - log_post_star
    log_evidence   = float(np.log(params.p_C + 1e-300)) + log_lik_c1

    # Cross-check via the exact Gaussian sub-identity.
    lds_marg_direct   = lds_loglik_c1_given_zev(home_x, z_ev_star, theta_star, params)
    lds_marg_via_chib = log_lik_star + log_lds_prior(z_lds_star, params) - ord_zlds

    if verbose:
        print(f"    Chib: ord(z^EV)={ord_zev:+.2f}  ord(Θ)={ord_theta:+.2f}  "
              f"ord(z^LDS)={ord_zlds:+.2f}")
        print(f"    Chib: log p(x|C=1)={log_lik_c1:+.1f}  log p(x,C=1)={log_evidence:+.1f}")
        print(f"    Chib check: z^LDS-marg direct={lds_marg_direct:+.2f} vs "
              f"via-Chib={lds_marg_via_chib:+.2f}  "
              f"(Δ={lds_marg_via_chib - lds_marg_direct:+.3f})")

    return ChibResult(
        log_lik_star      = float(log_lik_star),
        log_prior_star    = float(log_prior_star),
        ord_zev           = float(ord_zev),
        ord_theta         = float(ord_theta),
        ord_zlds          = float(ord_zlds),
        log_lik_c1        = float(log_lik_c1),
        log_evidence      = float(log_evidence),
        lds_marg_direct   = float(lds_marg_direct),
        lds_marg_via_chib = float(lds_marg_via_chib),
    )


# ===========================================================================
# Section 5.  Combined per-home driver + model comparison
# ===========================================================================

def infer_home(
    home_x: np.ndarray,
    params: ModelParams,
    *,
    S_burn: int = 200,
    S: int = 500,
    rng: np.random.Generator | None = None,
    home_id: int = -1,
    verbose: bool = True,
    decision: str = "rb",
    retain_z_ev: bool = False,
) -> HomeResult:
    """Run both tracks for one home and compare the two evidences.

    `decision` selects which C=1 evidence estimate drives the hard C_hat / soft
    c_prob:
      - "rb"     — z^LDS-marginal estimate A′ (default; much less biased than the
                   plug-in joint, nearly free).
      - "plugin" — plug-in complete-data joint A (not comparable across C; see §5).
      - "chib"   — exact marginal B via the Chib estimator (runs the extra reduced
                   run; slowest).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    c0 = infer_home_c0(home_x, params, home_id=home_id, verbose=verbose)
    c1 = infer_home_c1(home_x, params, S_burn=S_burn, S=S, rng=rng,
                       home_id=home_id, verbose=verbose, retain_z_ev=retain_z_ev,
                       compute_chib=(decision == "chib"))

    log_ev_c0 = c0.log_evidence
    log_ev_c1 = {
        "rb":     c1.log_evidence_rb,
        "plugin": c1.log_evidence_plugin,
        "chib":   c1.log_evidence_chib,
    }[decision]
    C_hat = int(log_ev_c1 > log_ev_c0)

    if verbose:
        logden = logsumexp([log_ev_c0, log_ev_c1])
        p_c1   = float(np.exp(log_ev_c1 - logden))
        print(f"  [home {home_id}] COMPARE ({decision}): "
              f"log p(x,C=0)={log_ev_c0:+.1f}  log p(x,C=1)={log_ev_c1:+.1f}  "
              f"→ P(C=1)={p_c1:.4f}  Ĉ={C_hat}")

    return HomeResult(
        home_id         = home_id,
        c0              = c0,
        c1              = c1,
        C_hat           = C_hat,
        log_evidence_c0 = log_ev_c0,
        log_evidence_c1 = log_ev_c1,
        decision        = decision,
    )


def infer_all(
    df: pd.DataFrame,
    params: ModelParams,
    *,
    S_burn: int = 200,
    S: int = 500,
    seed: int = 0,
    decision: str = "rb",
    verbose: bool = True,
) -> dict[int, HomeResult]:
    """Run `infer_home` (both tracks + comparison) on every home in df."""
    if verbose:
        print("=" * 60)
        print("INFERENCE: C=0 (exact) + C=1 (Gibbs) per home, then compare")
        print("=" * 60)

    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    homes     = list(sorted_df["home_id"].unique())
    rng       = np.random.default_rng(seed)

    results: dict[int, HomeResult] = {}
    t0 = time.time()
    for i, hid in enumerate(homes):
        g = sorted_df[sorted_df["home_id"] == hid]
        D = len(g) // T
        x = g["total_load"].to_numpy().reshape(D, T).astype(np.float64)

        if verbose:
            true_c = int(g["has_ev"].iloc[0]) if "has_ev" in g.columns else "?"
            print(f"\n[{i+1}/{len(homes)}] home {hid}  D={D}  true_c={true_c}")

        results[int(hid)] = infer_home(
            x, params, S_burn=S_burn, S=S, rng=rng,
            home_id=int(hid), decision=decision, verbose=verbose,
        )

    if verbose:
        print(f"\nAll homes done in {time.time() - t0:.1f}s")

    return results


# ===========================================================================
# Section 6.  Heuristic adapter (comparison baseline, unchanged)
# ===========================================================================

def build_heuristic_homes(df: pd.DataFrame) -> dict:
    """Reconstruct {dataid: (has_car, city, df_with_load_col)} from flat train_df.

    Format expected by `notebooks/utils/first_diff_logistic` — the heuristic
    C-detector used as a comparison baseline in the inference notebook.
    """
    out = {}
    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    for hid, g in sorted_df.groupby("home_id", sort=False):
        out[int(hid)] = (
            bool(g["has_ev"].iloc[0]),
            g["city"].iloc[0],
            g[["total_load"]].rename(columns={"total_load": "load"}).reset_index(drop=True),
        )
    return out
