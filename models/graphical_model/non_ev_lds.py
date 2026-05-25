"""Non-EV submodel — per-home, per-day linear dynamical system.

For each home n the Non-EV observations evolve as a standard LDS where
the "time axis" is **days**, and the per-day state and observation are both
T-dimensional vectors:

    z^(n)_1     ~  N( mu_0, Sigma_0 )                    # initial latent
    z^(n)_d     ~  N( A z^(n)_{d-1}, Q )      d ≥ 2      # transition
    x^Non-EV_d  ~  N( C z^(n)_d,    R )                  # emission

The LDS parameters (A, C, Q, R, mu_0, Sigma_0) are GLOBAL — shared across
homes — and fit by EM. Only the latent sequence z^(n)_{1:D_n} is per-home.

Current default instantiation (specs/model.md):
    A = I_L                random-walk transition
    C = I_T                identity emission, L = T
    Q, R                   diagonal (enforced post-M-step)
    mu_0, Sigma_0          free parameters

The math is written for the general case (any L, any A, C, full Q, R). The
identity defaults plus diagonal Q, R are passed in at construction time, and
EM flags `fit_A`, `fit_C` let us later relax those constraints without a
rewrite.

Module layout:

    Section 1.  Data containers
                  LDSParams, FilterResult, SmoothResult, LDSSufficientStats
    Section 2.  Kalman recursions
                  kalman_filter, rts_smooth, kalman_sample (FFBS),
                  kalman_loglik   — all delegated to via LDSParams methods
    Section 3.  EM fitting
                  fit_lds_em (top-level), _em_e_step, _em_m_step
    Section 4.  Gibbs adapter
                  sample_z_lds — wrapper used by inference.py

Latents handled here:
    z^LDS,(n)_{d,t}                       (D_n, T) per home — Non-EV latent

Global params estimated here (returned in an LDSParams object):
    A, C, Q, R, mu_0, Sigma_0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Default latent dimension when none is specified. Matches T=96 in params.py
# (kept inline here to avoid a circular import with params.py, which holds
# LDSParams inside ModelParams).
_DEFAULT_LATENT_DIM = 96


# ===========================================================================
# Section 1 — Data containers
# ===========================================================================

# Floors for variance / covariance diagonals — keep matrices PSD numerically.
Q_DIAG_FLOOR = 1e-6
R_DIAG_FLOOR = 1e-6
SIGMA0_DIAG_FLOOR = 1e-6


@dataclass
class LDSParams:
    """All LDS parameters in one place, with attached Kalman methods.

    Shapes use L = latent_dim, M = obs_dim.

    For the current Non-EV setup, L = M = T (= 96).
    """

    A:       np.ndarray   # (L, L)   transition matrix
    C:       np.ndarray   # (M, L)   emission matrix
    Q:       np.ndarray   # (L, L)   transition covariance (PSD)
    R:       np.ndarray   # (M, M)   emission covariance (PSD)
    mu_0:    np.ndarray   # (L,)     initial mean
    Sigma_0: np.ndarray   # (L, L)   initial covariance (PSD)

    # ---- factory ---------------------------------------------------------

    @staticmethod
    def identity_diagonal(
        latent_dim: int = _DEFAULT_LATENT_DIM,
        obs_dim:    int = _DEFAULT_LATENT_DIM,
        q_diag:     float = 1.0,
        r_diag:     float = 1.0,
        sigma0_diag: float = 1.0,
        mu_0_value: float = 0.0,
    ) -> "LDSParams":
        """Build a fresh `LDSParams` with A=I, C=I and diagonal Q, R, Sigma_0.

        Useful as an EM warm start; concrete diagonals get fit by EM.
        """
        if latent_dim != obs_dim:
            raise ValueError("identity_diagonal requires latent_dim == obs_dim")
        L = latent_dim
        return LDSParams(
            A       = np.eye(L),
            C       = np.eye(L),
            Q       = np.eye(L) * q_diag,
            R       = np.eye(L) * r_diag,
            mu_0    = np.full(L, mu_0_value, dtype=np.float64),
            Sigma_0 = np.eye(L) * sigma0_diag,
        )

    # ---- shape sanity ---------------------------------------------------

    @property
    def latent_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.C.shape[0])

    def diag_R(self) -> np.ndarray:
        """Per-t diagonal of R, treated as the per-t emission variance.
        Used as the σ²-equivalent input to the HMM forward pass (see inference.py)."""
        return np.diag(self.R).copy()

    # ---- Kalman methods (thin wrappers on the module-level functions) -----

    def filter(self, obs, extra_obs_cov=None) -> "FilterResult":
        return kalman_filter(self, obs, extra_obs_cov=extra_obs_cov)

    def smooth(self, obs, extra_obs_cov=None) -> "SmoothResult":
        return rts_smooth(self, kalman_filter(self, obs, extra_obs_cov=extra_obs_cov))

    def sample(self, obs, rng, extra_obs_cov=None) -> np.ndarray:
        return kalman_sample(self, obs, rng, extra_obs_cov=extra_obs_cov)

    def loglik(self, obs, extra_obs_cov=None) -> float:
        return kalman_filter(self, obs, extra_obs_cov=extra_obs_cov).log_lik

    # ---- printing --------------------------------------------------------

    def summary(self) -> str:
        L = self.latent_dim
        M = self.obs_dim
        Q_diag = np.diag(self.Q)
        R_diag = np.diag(self.R)
        S0_diag = np.diag(self.Sigma_0)
        lines = [
            "LDSParams summary",
            "-" * 40,
            f"  latent_dim L = {L}, obs_dim M = {M}",
            f"  A             shape={self.A.shape}  (identity? {np.allclose(self.A, np.eye(L))})",
            f"  C             shape={self.C.shape}  (identity? "
            f"{L == M and np.allclose(self.C, np.eye(L))})",
            f"  mu_0          (min={self.mu_0.min():+.3f}, "
            f"median={np.median(self.mu_0):+.3f}, max={self.mu_0.max():+.3f})",
            f"  diag(Sigma_0) (min={S0_diag.min():.4f}, "
            f"median={np.median(S0_diag):.4f}, max={S0_diag.max():.4f})",
            f"  diag(Q)       (min={Q_diag.min():.4f}, "
            f"median={np.median(Q_diag):.4f}, max={Q_diag.max():.4f})",
            f"  diag(R)       (min={R_diag.min():.4f}, "
            f"median={np.median(R_diag):.4f}, max={R_diag.max():.4f})",
        ]
        return "\n".join(lines)


@dataclass
class FilterResult:
    """Output of the Kalman forward pass.

    All shapes per-home (single sequence of length D):
        z_pred  : (D, L)        predict-step mean   E[z_d | x_{1:d-1}]
        P_pred  : (D, L, L)     predict-step cov    Cov[z_d | x_{1:d-1}]
        z_filt  : (D, L)        filter-step mean    E[z_d | x_{1:d}]
        P_filt  : (D, L, L)     filter-step cov     Cov[z_d | x_{1:d}]
        log_lik : float         sum_d log p(x_d | x_{1:d-1})   — marginal log-likelihood
    """
    z_pred:  np.ndarray
    P_pred:  np.ndarray
    z_filt:  np.ndarray
    P_filt:  np.ndarray
    log_lik: float


@dataclass
class SmoothResult:
    """Output of the RTS smoother.

        z_smooth   : (D, L)        E[z_d | x_{1:D}]
        P_smooth   : (D, L, L)     Cov[z_d | x_{1:D}]
        P_lag1     : (D, L, L)     Cov[z_d, z_{d-1} | x_{1:D}] for d=1..D-1.
                                   P_lag1[0] is unused (zeros).
        log_lik    : float         passed through from the forward pass
    """
    z_smooth: np.ndarray
    P_smooth: np.ndarray
    P_lag1:   np.ndarray
    log_lik:  float


@dataclass
class LDSSufficientStats:
    """Sufficient statistics aggregated across all homes for one EM iter.

    See _em_m_step() for how each enters the M-step formulas.
    """
    N_homes:    int                 # number of training homes
    N_total:    int                 # sum_n D_n         (rows of x across homes)
    N_pair:     int                 # sum_n (D_n - 1)   (transition counts)
    S_z0:       np.ndarray          # (L,)         sum_n E[z_{n,0} | x_n]
    S_zz0:      np.ndarray          # (L, L)       sum_n E[z_{n,0} z_{n,0}^T | x_n]
    S_zz_curr:  np.ndarray          # (L, L)       sum_{n, d≥1} E[z_{n,d} z_{n,d}^T | x_n]
    S_zz_prev:  np.ndarray          # (L, L)       sum_{n, d≥1} E[z_{n,d-1} z_{n,d-1}^T | x_n]
    S_zz_pair:  np.ndarray          # (L, L)       sum_{n, d≥1} E[z_{n,d} z_{n,d-1}^T | x_n]
    S_zz_all:   np.ndarray          # (L, L)       sum_{n, d} E[z_{n,d} z_{n,d}^T | x_n]   (all d)
    S_x_z:      np.ndarray          # (M, L)       sum_{n, d} x_{n,d} E[z_{n,d}^T | x_n]
    S_xx:       np.ndarray          # (M, M)       sum_{n, d} x_{n,d} x_{n,d}^T
    total_log_lik: float            # sum_n log p(x_n) under current params


# ===========================================================================
# Section 2 — Kalman recursions (general LDS)
# ===========================================================================

def kalman_filter(
    params: LDSParams,
    obs:    np.ndarray,                   # (D, M)
    *,
    extra_obs_cov: np.ndarray | None = None,   # (D, M) — added to diag(R) per d, or None
) -> FilterResult:
    """Standard Kalman forward pass.

    `extra_obs_cov[d]` is an additional per-t variance ADDED to the diagonal of R
    on day d (used when residuals x - Θ_z have heteroscedastic obs noise in the
    Gibbs sampler — see sample_z_lds). When None, the obs cov is just R.
    """
    A, C, Q, R       = params.A, params.C, params.Q, params.R
    mu_0, Sigma_0    = params.mu_0, params.Sigma_0
    D, M             = obs.shape
    L                = params.latent_dim

    z_pred  = np.empty((D, L),    dtype=np.float64)
    P_pred  = np.empty((D, L, L), dtype=np.float64)
    z_filt  = np.empty((D, L),    dtype=np.float64)
    P_filt  = np.empty((D, L, L), dtype=np.float64)
    log_lik = 0.0
    I_L     = np.eye(L)

    for d in range(D):
        # --- predict step ------------------------------------------------
        if d == 0:
            z_pred[d] = mu_0
            P_pred[d] = Sigma_0
        else:
            z_pred[d] = A @ z_filt[d-1]
            P_pred[d] = A @ P_filt[d-1] @ A.T + Q

        # --- obs covariance on day d (R + per-cell extra noise) -----------
        obs_cov_d = R if extra_obs_cov is None else (R + np.diag(extra_obs_cov[d]))

        # --- update step (innovation form) -------------------------------
        innov = obs[d] - C @ z_pred[d]                # (M,)
        S     = C @ P_pred[d] @ C.T + obs_cov_d       # (M, M) innovation cov
        # Solve K = P_pred C^T S^{-1}  via Cholesky of S for stability
        L_S   = np.linalg.cholesky(S)
        K     = np.linalg.solve(L_S.T, np.linalg.solve(L_S, C @ P_pred[d])).T   # (L, M)

        z_filt[d] = z_pred[d] + K @ innov
        # Joseph form for numerical-stability-friendly covariance update
        I_KC      = I_L - K @ C
        P_filt[d] = I_KC @ P_pred[d] @ I_KC.T + K @ obs_cov_d @ K.T
        # Symmetrize against drift
        P_filt[d] = 0.5 * (P_filt[d] + P_filt[d].T)

        # --- contribute to marginal log-likelihood -----------------------
        log_lik += _gaussian_loglik_chol(innov, L_S)

    return FilterResult(z_pred=z_pred, P_pred=P_pred,
                        z_filt=z_filt, P_filt=P_filt,
                        log_lik=log_lik)


def rts_smooth(params: LDSParams, fr: FilterResult) -> SmoothResult:
    """RTS backward smoother. Also returns lag-1 smoothed covariances for EM.

    Lag-1 smoothed covariance is Cov[z_d, z_{d-1} | x_{1:D}] = J_{d-1} P_smooth[d].
    Stored at index d (for d=1..D-1); P_lag1[0] is zero.
    """
    A = params.A
    D, L = fr.z_filt.shape

    z_smooth = np.empty_like(fr.z_filt)
    P_smooth = np.empty_like(fr.P_filt)
    P_lag1   = np.zeros_like(fr.P_filt)
    J_store  = np.zeros((D, L, L), dtype=np.float64)

    # Initialise at the last step
    z_smooth[D-1] = fr.z_filt[D-1]
    P_smooth[D-1] = fr.P_filt[D-1]

    for d in range(D - 2, -1, -1):
        # Smoother gain J_d = P_filt[d] A^T (P_pred[d+1])^{-1}
        P_pred_next = fr.P_pred[d+1]
        J = fr.P_filt[d] @ A.T @ np.linalg.inv(P_pred_next)
        J_store[d] = J

        z_smooth[d] = fr.z_filt[d] + J @ (z_smooth[d+1] - fr.z_pred[d+1])
        P_smooth[d] = fr.P_filt[d] + J @ (P_smooth[d+1] - P_pred_next) @ J.T
        P_smooth[d] = 0.5 * (P_smooth[d] + P_smooth[d].T)

    # Lag-1 covariances Cov[z_d, z_{d-1} | x_{1:D}] = J_{d-1} P_smooth[d]
    for d in range(1, D):
        P_lag1[d] = J_store[d-1] @ P_smooth[d]

    return SmoothResult(z_smooth=z_smooth, P_smooth=P_smooth,
                        P_lag1=P_lag1, log_lik=fr.log_lik)


def kalman_sample(
    params: LDSParams,
    obs:    np.ndarray,
    rng,
    *,
    extra_obs_cov: np.ndarray | None = None,
) -> np.ndarray:
    """FFBS for the LDS: forward filter, then backward-sample z_{1:D} as one joint draw.

    Returns z_sample : (D, L).
    """
    fr   = kalman_filter(params, obs, extra_obs_cov=extra_obs_cov)
    A    = params.A
    D, L = fr.z_filt.shape

    z_sample = np.empty((D, L), dtype=np.float64)

    # Sample z_{D-1} from N(z_filt[D-1], P_filt[D-1])
    z_sample[D-1] = _sample_mvn(fr.z_filt[D-1], fr.P_filt[D-1], rng)

    for d in range(D - 2, -1, -1):
        # p(z_d | z_{d+1}, x_{1:D}) is Gaussian with
        #   m_cond = z_filt[d] + J_d (z_sample[d+1] - z_pred[d+1])
        #   V_cond = P_filt[d] - J_d P_pred[d+1] J_d^T
        P_pred_next = fr.P_pred[d+1]
        J = fr.P_filt[d] @ A.T @ np.linalg.inv(P_pred_next)

        m_cond = fr.z_filt[d] + J @ (z_sample[d+1] - fr.z_pred[d+1])
        V_cond = fr.P_filt[d] - J @ P_pred_next @ J.T
        V_cond = 0.5 * (V_cond + V_cond.T)

        z_sample[d] = _sample_mvn(m_cond, V_cond, rng)

    return z_sample


def _sample_mvn(mean: np.ndarray, cov: np.ndarray, rng) -> np.ndarray:
    """Draw one sample from N(mean, cov). Tolerates mildly-PSD inputs via jitter."""
    L = mean.shape[0]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.eye(L)
        chol = np.linalg.cholesky(cov + jitter)
    return mean + chol @ rng.standard_normal(L)


def _gaussian_loglik_chol(innov: np.ndarray, L_S: np.ndarray) -> float:
    """log N(innov ; 0, S) given the Cholesky factor L_S of S (S = L_S L_S^T)."""
    M = innov.shape[0]
    # solve L_S y = innov  →  y^T y = innov^T S^{-1} innov
    y = np.linalg.solve(L_S, innov)
    return -0.5 * (M * np.log(2 * np.pi)
                   + 2 * np.log(np.diag(L_S)).sum()
                   + float(y @ y))


# ===========================================================================
# Section 3 — EM fitting
# ===========================================================================

def fit_lds_em(
    observations_per_home: list[np.ndarray],   # each (D_n, M); typically x_nev
    *,
    init: LDSParams | None = None,
    latent_dim: int = _DEFAULT_LATENT_DIM,
    fit_A: bool = False,
    fit_C: bool = False,
    diagonal_Q: bool = True,
    diagonal_R: bool = True,
    diagonal_Sigma_0: bool = True,
    max_iters: int = 50,
    tol: float = 1e-4,
    verbose: bool = True,
) -> LDSParams:
    """EM for the LDS parameters from per-home Non-EV observation sequences.

    Inputs
    ------
    observations_per_home  : list of (D_n, M) arrays of x^Non-EV per home
    init                   : optional warm-start LDSParams; defaults to identity_diagonal
    fit_A, fit_C           : whether the M-step updates A, C (defaults: held at identity)
    diagonal_Q/R/Sigma_0   : if True, project the M-step result to its diagonal

    EM loop:
      E-step (_em_e_step)  — for each home, run smoother and accumulate suff. stats
      M-step (_em_m_step)  — closed-form updates for the requested parameters

    Convergence: |Δ log L| / |log L| < tol  OR  max_iters reached.
    """
    if verbose:
        print("=" * 60)
        print("FIT: LDS via EM")
        print("=" * 60)
        N = len(observations_per_home)
        D_total = sum(o.shape[0] for o in observations_per_home)
        M = observations_per_home[0].shape[1]
        print(f"  N_homes={N}, total_days={D_total}, obs_dim={M}, latent_dim={latent_dim}")
        print(f"  fit_A={fit_A}, fit_C={fit_C}, "
              f"diagonal(Q/R/Σ₀)=({diagonal_Q},{diagonal_R},{diagonal_Sigma_0})")

    # ---- initialise -----------------------------------------------------
    if init is None:
        obs_dim = observations_per_home[0].shape[1]
        params  = LDSParams.identity_diagonal(latent_dim=latent_dim, obs_dim=obs_dim)
    else:
        params = init

    prev_ll = -np.inf
    t_start = time.time()
    for it in range(max_iters):
        # E-step: smooth each home, accumulate sufficient statistics
        ss = _em_e_step(params, observations_per_home)

        # M-step: closed-form param updates
        params = _em_m_step(
            ss, params,
            fit_A=fit_A, fit_C=fit_C,
            diagonal_Q=diagonal_Q, diagonal_R=diagonal_R,
            diagonal_Sigma_0=diagonal_Sigma_0,
        )

        delta = ss.total_log_lik - prev_ll
        if verbose:
            print(f"  iter {it:3d}:  logL = {ss.total_log_lik:.3f}  Δ = {delta:+.3e}  "
                  f"  (Q̄ diag = {np.diag(params.Q).mean():.4f}, "
                  f"R̄ diag = {np.diag(params.R).mean():.4f})")

        # Convergence test
        rel_delta = abs(delta) / max(abs(ss.total_log_lik), 1.0)
        if it > 0 and rel_delta < tol:
            if verbose:
                print(f"  EM converged at iter {it} (Δrel = {rel_delta:.2e})")
            break
        prev_ll = ss.total_log_lik

    if verbose:
        print(f"  EM done in {time.time() - t_start:.1f}s")
        print("\n" + params.summary())

    return params


def _em_e_step(
    params: LDSParams,
    observations_per_home: list[np.ndarray],
) -> LDSSufficientStats:
    """E-step: for each home, smooth and accumulate sufficient statistics.

    Statistics aggregated (see LDSSufficientStats docstring for each):
      S_z0, S_zz0      — initial-state stats (for mu_0, Sigma_0)
      S_zz_*           — transition stats   (for A, Q)
      S_x_z, S_xx, S_zz_all — emission stats (for C, R)
    """
    L = params.latent_dim
    M = params.obs_dim

    S_z0       = np.zeros(L,         dtype=np.float64)
    S_zz0      = np.zeros((L, L),    dtype=np.float64)
    S_zz_curr  = np.zeros((L, L),    dtype=np.float64)
    S_zz_prev  = np.zeros((L, L),    dtype=np.float64)
    S_zz_pair  = np.zeros((L, L),    dtype=np.float64)
    S_zz_all   = np.zeros((L, L),    dtype=np.float64)
    S_x_z      = np.zeros((M, L),    dtype=np.float64)
    S_xx       = np.zeros((M, M),    dtype=np.float64)
    total_log_lik = 0.0
    N_total = 0
    N_pair  = 0

    for x in observations_per_home:
        D = x.shape[0]
        sm = params.smooth(x)                            # E-step kernel: smooth this home
        total_log_lik += sm.log_lik

        # Per-d  E[z_d z_d^T | x_n]  =  P_smooth[d] + z_smooth[d] z_smooth[d]^T
        Ezz = sm.P_smooth + np.einsum("di,dj->dij", sm.z_smooth, sm.z_smooth)   # (D, L, L)
        # Per-d  E[z_d z_{d-1}^T | x_n] = P_lag1[d] + z_smooth[d] z_smooth[d-1]^T   for d≥1
        Ezz_lag = sm.P_lag1 + np.einsum("di,dj->dij", sm.z_smooth,
                                                       np.vstack([np.zeros((1, L)),
                                                                  sm.z_smooth[:-1]]))

        # Initial-state stats from d=0
        S_z0  += sm.z_smooth[0]
        S_zz0 += Ezz[0]

        # Transition stats from d=1..D-1
        if D >= 2:
            S_zz_curr += Ezz[1:].sum(axis=0)
            S_zz_prev += Ezz[:-1].sum(axis=0)
            S_zz_pair += Ezz_lag[1:].sum(axis=0)
            N_pair    += (D - 1)

        # Emission stats over all d
        S_zz_all += Ezz.sum(axis=0)
        S_x_z    += x.T @ sm.z_smooth                    # (M, D) (D, L) = (M, L)
        S_xx     += x.T @ x                              # (M, M)
        N_total  += D

    return LDSSufficientStats(
        N_homes=len(observations_per_home),
        N_total=N_total, N_pair=N_pair,
        S_z0=S_z0, S_zz0=S_zz0,
        S_zz_curr=S_zz_curr, S_zz_prev=S_zz_prev, S_zz_pair=S_zz_pair,
        S_zz_all=S_zz_all,
        S_x_z=S_x_z, S_xx=S_xx,
        total_log_lik=total_log_lik,
    )


def _em_m_step(
    ss:     LDSSufficientStats,
    params: LDSParams,
    *,
    fit_A: bool,
    fit_C: bool,
    diagonal_Q: bool,
    diagonal_R: bool,
    diagonal_Sigma_0: bool,
) -> LDSParams:
    """M-step: closed-form updates for whichever parameters are flagged on.

    For parameters held fixed (A or C), we still update Q (resp. R) using the
    held-fixed value plugged into the standard quadratic-form residual.
    """
    N = ss.N_homes

    # --- mu_0 and Sigma_0 (always update) --------------------------------
    mu_0_new = ss.S_z0 / N
    Sigma_0_new = ss.S_zz0 / N - np.outer(mu_0_new, mu_0_new)
    Sigma_0_new = _project_psd(Sigma_0_new, diagonal=diagonal_Sigma_0, floor=SIGMA0_DIAG_FLOOR)

    # --- A (optional) ----------------------------------------------------
    A_new = params.A
    if fit_A and ss.N_pair > 0:
        A_new = ss.S_zz_pair @ np.linalg.inv(ss.S_zz_prev)

    # --- Q (always update; uses current or updated A) --------------------
    if ss.N_pair > 0:
        Q_residual = (
            ss.S_zz_curr
            - A_new @ ss.S_zz_pair.T
            - ss.S_zz_pair @ A_new.T
            + A_new @ ss.S_zz_prev @ A_new.T
        )
        Q_new = Q_residual / ss.N_pair
        Q_new = _project_psd(Q_new, diagonal=diagonal_Q, floor=Q_DIAG_FLOOR)
    else:
        Q_new = params.Q   # no transitions observed (all homes are length 1)

    # --- C (optional) ----------------------------------------------------
    C_new = params.C
    if fit_C and ss.N_total > 0:
        C_new = ss.S_x_z @ np.linalg.inv(ss.S_zz_all)

    # --- R (always update; uses current or updated C) --------------------
    if ss.N_total > 0:
        R_residual = (
            ss.S_xx
            - C_new @ ss.S_x_z.T
            - ss.S_x_z @ C_new.T
            + C_new @ ss.S_zz_all @ C_new.T
        )
        R_new = R_residual / ss.N_total
        R_new = _project_psd(R_new, diagonal=diagonal_R, floor=R_DIAG_FLOOR)
    else:
        R_new = params.R

    return LDSParams(
        A       = A_new,
        C       = C_new,
        Q       = Q_new,
        R       = R_new,
        mu_0    = mu_0_new,
        Sigma_0 = Sigma_0_new,
    )


def _project_psd(M: np.ndarray, *, diagonal: bool, floor: float) -> np.ndarray:
    """Project an estimated covariance onto the requested structure.

    diagonal=True  → drop off-diagonals; floor the diagonal at `floor`.
    diagonal=False → symmetrize; floor any negative eigenvalues at `floor`.
    """
    if diagonal:
        diag = np.maximum(np.diag(M), floor)
        return np.diag(diag)
    # full case: symmetrize + floor eigenvalues
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ===========================================================================
# Section 4 — Gibbs adapter (used by inference.py)
# ===========================================================================

def sample_z_lds(
    x:             np.ndarray,                # (D, T)   observed total grid power
    z_ev:          np.ndarray,                # (D, T)   current charging-state sample
    theta:         np.ndarray,                # (K,)     current charging magnitudes
    sigma2_ev:     np.ndarray,                # (K,)     EV emission variance per state
    lds:           LDSParams,
    rng,
) -> np.ndarray:
    """Gibbs block: sample z^LDS_{1:D} from its conditional posterior given the
    EV-side latents.

    Conditional on (z_ev, theta), the residual x - theta[z_ev] is a noisy
    observation of C z^LDS_d with **heteroscedastic** per-cell noise:

        residual[d,t]  =  x[d,t] - theta[z_ev[d,t]]
                       =  (C z^LDS_d)[t]  +  (epsilon_EV)[d,t]  +  v_d[t]
        var(residual[d,t] | z_lds)  =  sigma2_ev[z_ev[d,t]] + R[t,t]

    We use the standard LDS FFBS, passing the per-cell EV variance as
    `extra_obs_cov` so each day's effective obs covariance is R + diag(σ²_EV).
    Off-diagonals of R remain unchanged.
    """
    residual      = x - theta[z_ev]                  # (D, T)
    extra_obs_cov = sigma2_ev[z_ev]                  # (D, T) — per-cell EV noise added to diag(R)
    return lds.sample(residual, rng, extra_obs_cov=extra_obs_cov)
