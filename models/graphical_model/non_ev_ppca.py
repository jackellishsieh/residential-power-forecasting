"""Non-EV submodel — hierarchical PPCA mean + (global | hierarchical) variance.

Implements specs/model.md §2.1–§2.6 (current model). To be replaced in Stage 2
by a per-home daily LDS — see specs/model.md (forthcoming §2).

Split into two sections:
    1.  Fit-time     fit_background  (dispatches on omega_mode)
    2.  Sample-time  sample_eta, sample_omega, compute_sigma_eta_inv

Latents handled here:
    eta^(n)_t         — per-home Non-EV mean profile (T-vector)
    (omega^(n)_t)^2   — per-home Non-EV variance profile (only when hierarchical)

Global params estimated here:
    eta_bar, W_eta, psi_eta                  — PPCA prior on eta
    sigma2_nev_global                        — only when omega_mode == "global"
    a_omega, b_omega                         — only when omega_mode == "hierarchical"
"""

from __future__ import annotations

import numpy as np

from .params import (
    IG_MIN_SHAPE, OMEGA2_FLOOR, PSI_FLOOR,
    SLICE_MAX_SHRINK, SLICE_MAX_STEPS, SLICE_W,
    ModelParams,
)


# ===========================================================================
# 1. FIT-TIME — eta-prior and omega-parameterization
# ===========================================================================

def fit_background(
    home_arrays: dict,
    homes: list[int],
    *,
    ppca_rank: int,
    omega_mode: str,
    verbose: bool,
):
    """Fit the Non-EV submodel from labeled training data.

    Mean side (eta) is hierarchical PPCA regardless of omega_mode.
    Variance side dispatches on omega_mode:
        "global"        → returns (sigma2_nev_global, a_omega=None, b_omega=None)
        "hierarchical"  → returns (sigma2_nev_global=None, a_omega, b_omega)

    Returns
    -------
    eta_bar           : (T,)   global mean profile
    W_eta             : (T, r) PPCA loading matrix
    psi_eta           : (T,)   per-t residual variance
    sigma2_nev_global : (T,) or None
    a_omega           : (T,) or None
    b_omega           : (T,) or None
    """
    # --- 3a. Per-home plug-ins: empirical day-mean and empirical day-variance --
    eta_hat    = np.stack([home_arrays[hid]["x_nev"].mean(axis=0) for hid in homes])
    omega2_hat = np.stack([home_arrays[hid]["x_nev"].var(axis=0, ddof=0)
                           for hid in homes])
    Ds = np.array([home_arrays[hid]["D"] for hid in homes], dtype=np.float64)

    if verbose:
        print(f"  Per-home plug-ins:")
        print(f"    eta_hat^(n)_t   range over (n,t): "
              f"min={eta_hat.min():+.3f}, max={eta_hat.max():+.3f}, "
              f"mean over n,t={eta_hat.mean():+.3f}")
        sig_hat = np.sqrt(omega2_hat)
        print(f"    omega_hat^(n)_t (std-dev) range : "
              f"min={sig_hat.min():.3f}, median={np.median(sig_hat):.3f}, "
              f"max={sig_hat.max():.3f}")

    # --- 3b. eta-prior fit (hierarchical PPCA — always) -----------------------
    eta_bar, W_eta, psi_eta = _fit_eta_prior(
        eta_hat, omega2_hat, Ds, ppca_rank=ppca_rank, verbose=verbose,
    )

    # --- 3c. omega-parameterization (DISPATCH on omega_mode) ------------------
    if omega_mode == "global":
        sigma2_nev_global = _fit_omega_global(omega2_hat, Ds, verbose=verbose)
        a_omega = b_omega = None
    elif omega_mode == "hierarchical":
        a_omega, b_omega = _fit_omega_prior(omega2_hat, verbose=verbose)
        sigma2_nev_global = None
    else:
        raise ValueError(f"unknown omega_mode={omega_mode!r}; "
                         f"expected 'global' or 'hierarchical'")

    return eta_bar, W_eta, psi_eta, sigma2_nev_global, a_omega, b_omega


def _fit_eta_prior(
    eta_hat: np.ndarray,        # (N, T) per-home empirical mean profile
    omega2_hat: np.ndarray,     # (N, T) per-home empirical variance profile
    Ds: np.ndarray,             # (N,)   per-home day counts
    *,
    ppca_rank: int,
    verbose: bool,
):
    """eta-prior fit via mean + bias-corrected truncated-eigen factor analysis.

    Sigma_eta = W W^T + diag(psi)  where W in R^{T x r} and psi in R^T_>0.

    The empirical sample covariance is rank-deficient (rank <= N-1 < T at
    N=50, T=96), so we *cannot* use the full sample covariance as a prior
    precision. We truncate to the top r eigenvalues and absorb the rest into
    a diagonal residual.

    Bias correction: Var_n(eta_hat^(n)_t) overestimates the true cross-home
    variance by within-home noise (omega_hat^(n)_t)^2 / D^(n). We subtract
    this from the diagonal of the sample covariance before eigendecomp.

    NOTE: heuristic low-rank-plus-diagonal decomposition, not the MLE for
    factor analysis (which would need EM). At N=50 it's a close approximation.
    """
    N, T_ = eta_hat.shape
    eta_bar = eta_hat.mean(axis=0)
    centered = eta_hat - eta_bar

    # Sample covariance, bias-subtracted on the diagonal
    S_emp     = (centered.T @ centered) / max(N - 1, 1)
    bias_diag = np.mean(omega2_hat / Ds[:, None], axis=0)
    S_corr    = S_emp.copy()
    np.fill_diagonal(S_corr, np.diag(S_corr) - bias_diag)
    S_corr    = 0.5 * (S_corr + S_corr.T)                # symmetrise against numerical drift

    # Truncated eigendecomp (descending eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(S_corr)
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    r = int(ppca_rank)
    if r > 0:
        top_eigvals = np.maximum(eigvals[:r], 0.0)
        W_eta = eigvecs[:, :r] * np.sqrt(top_eigvals)[None, :]
        residual = S_corr - W_eta @ W_eta.T
        psi_eta  = np.maximum(np.diag(residual), PSI_FLOOR)
    else:
        W_eta   = np.zeros((T_, 0), dtype=np.float64)
        psi_eta = np.maximum(np.diag(S_corr), PSI_FLOOR)

    if verbose:
        total_var = float(np.maximum(eigvals, 0.0).sum())
        topr_var  = float(np.maximum(eigvals[:max(r, 1)], 0.0).sum()) if r > 0 else 0.0
        frac = topr_var / max(total_var, 1e-12)
        print(f"  Eta-prior fit:")
        print(f"    eta_bar         min={eta_bar.min():+.3f} median="
              f"{np.median(eta_bar):+.3f} max={eta_bar.max():+.3f}")
        print(f"    top-5 eigenvalues of S_corr: "
              f"{np.array2string(eigvals[:5], precision=3)}")
        print(f"    PPCA rank r = {r}; variance explained by top {r} factors: "
              f"{frac:.3f}")
        print(f"    psi_eta (per-t residual variance): "
              f"min={psi_eta.min():.4f}, median={np.median(psi_eta):.4f}, "
              f"max={psi_eta.max():.4f}")
        neg_eig = int((eigvals < 0).sum())
        if neg_eig > 0:
            print(f"    NOTE: bias-corrected S_corr had {neg_eig} negative "
                  f"eigenvalues (floored to 0 in W).")

    return eta_bar, W_eta, psi_eta


def _fit_omega_global(
    omega2_hat: np.ndarray,    # (N, T)
    Ds: np.ndarray,            # (N,)
    *,
    verbose: bool,
) -> np.ndarray:
    """Global per-t Non-EV variance, weighted by per-home day counts (specs §2.7.4)."""
    weights = Ds / Ds.sum()                                # (N,)
    sigma2 = (weights[:, None] * omega2_hat).sum(axis=0)   # (T,)

    if verbose:
        sig = np.sqrt(sigma2)
        print(f"  Omega-fit (global, fixed at inference):")
        print(f"    sigma_nev_global_t (std-dev): "
              f"min={sig.min():.3f}, median={np.median(sig):.3f}, "
              f"max={sig.max():.3f}")
    return sigma2


def _fit_omega_prior(
    omega2_hat: np.ndarray,     # (N, T)
    *,
    verbose: bool,
):
    """Method-of-moments per t for the InvGamma prior (specs/model.md §2.4).

        a = m^2 / v + 2,    b = m * (a - 1)

    Floors a at IG_MIN_SHAPE to keep IG variance finite.
    """
    m_omega2 = omega2_hat.mean(axis=0)
    v_omega2 = omega2_hat.var(axis=0, ddof=1)

    a_omega = np.maximum(m_omega2 ** 2 / np.maximum(v_omega2, 1e-12) + 2.0,
                          IG_MIN_SHAPE)
    b_omega = m_omega2 * (a_omega - 1.0)

    if verbose:
        prior_mean      = b_omega / (a_omega - 1.0)
        prior_std_ratio = np.sqrt(1.0 / np.maximum(a_omega - 2.0, 1e-12))
        print(f"  Omega-prior fit (MoM per t):")
        print(f"    a_omega: min={a_omega.min():.2f}, median={np.median(a_omega):.2f}, "
              f"max={a_omega.max():.2f}")
        print(f"    b_omega: min={b_omega.min():.4f}, median={np.median(b_omega):.4f}, "
              f"max={b_omega.max():.4f}")
        print(f"    E[(omega_t)^2] prior: min={prior_mean.min():.4f}, "
              f"median={np.median(prior_mean):.4f}, max={prior_mean.max():.4f}")
        print(f"    coefficient-of-variation of prior: "
              f"min={prior_std_ratio.min():.3f}, max={prior_std_ratio.max():.3f}")

    return a_omega, b_omega


# ===========================================================================
# 2. SAMPLE-TIME — Gibbs blocks
# ===========================================================================

# --- Block 3: eta  (T-dim conjugate Gaussian under PPCA prior) ----------------

def compute_sigma_eta_inv(W: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Sigma_eta^{-1} = (W W^T + diag(psi))^{-1} via Woodbury.

        = diag(1/psi) - diag(1/psi) W (I_r + W^T diag(1/psi) W)^{-1} W^T diag(1/psi)

    Cached once per infer_home call since W, psi don't change during a chain.
    """
    r       = W.shape[1]
    inv_psi = 1.0 / psi
    if r == 0:
        return np.diag(inv_psi)
    M  = np.eye(r) + W.T @ (inv_psi[:, None] * W)                  # (r, r)
    WP = inv_psi[:, None] * W                                       # (T, r)
    return np.diag(inv_psi) - WP @ np.linalg.solve(M, WP.T)


def sample_eta(
    x:      np.ndarray,    # (D, T)
    z:      np.ndarray,    # (D, T)
    theta:  np.ndarray,    # (K,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
    Sigma_eta_inv:           np.ndarray,   # (T, T) cached
    Sigma_eta_inv_etabar:    np.ndarray,   # (T,)   cached
    rng,
) -> np.ndarray:
    """T-dim conjugate Gaussian sample for eta under PPCA prior (specs §2.1).

    Likelihood: x[d,t] - theta[z[d,t]]  ~  N( eta[t], sigma2_ev[z[d,t]] + omega2[t] )
                                          (heteroscedastic in (d,t))

    Posterior precision: Sigma_eta^{-1} + diag(lambda_t)
                         where lambda_t = sum_d 1/(sigma2_ev[z[d,t]] + omega2[t])
    Posterior mean:      Sigma_post · (Sigma_eta^{-1} eta_bar + h_data),
                         h_data[t] = sum_d (x[d,t] - theta[z[d,t]]) / var[d,t]
    """
    D, T_ = x.shape

    var_dt     = params.sigma2_ev[z] + omega2[None, :]            # (D, T)
    inv_var_dt = 1.0 / var_dt

    lambda_t = inv_var_dt.sum(axis=0)                              # (T,)
    h_data   = ((x - theta[z]) * inv_var_dt).sum(axis=0)           # (T,)

    # Posterior precision Λ = Σ_η^{-1} + diag(λ_t)
    Lambda = Sigma_eta_inv.copy()
    Lambda.flat[::T_ + 1] += lambda_t

    # Info vector h = Σ_η^{-1} eta_bar + h_data
    h = Sigma_eta_inv_etabar + h_data

    # Solve Λ μ = h, then sample η = μ + L^{-T} ξ   (Λ = L L^T)
    L  = np.linalg.cholesky(Lambda)
    mu = np.linalg.solve(L.T, np.linalg.solve(L, h))
    xi = rng.standard_normal(T_)
    return mu + np.linalg.solve(L.T, xi)


# --- Block 4: omega^2  (univariate slice sample per t, hierarchical only) -----

def sample_omega(
    x:      np.ndarray,    # (D, T)
    z:      np.ndarray,    # (D, T)
    theta:  np.ndarray,    # (K,)
    eta:    np.ndarray,    # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
    rng,
):
    """Slice-sample (omega_t)^2 for each t independently in log-variance space.

    Posterior per t (with l = log(omega^2)):

        log p(l | rest) = log p_IG(exp(l); a_omega_t, b_omega_t) + l   (Jacobian)
                       - 0.5 Σ_d [ log(sigma2_ev[z[d,t]] + exp(l))
                                    + (x[d,t] - theta[z[d,t]] - eta[t])^2
                                      / (sigma2_ev[z[d,t]] + exp(l)) ]

    Returns (new_omega2 (T,), total_log_density_evals (int)).
    """
    D, T_ = x.shape
    new_omega2  = omega2.copy()
    total_evals = 0

    sigma2_ev_dt = params.sigma2_ev[z]                # (D, T)
    resid        = x - theta[z] - eta[None, :]        # (D, T)
    resid_sq     = resid ** 2

    a_omega = params.a_omega
    b_omega = params.b_omega

    for t in range(T_):
        sig2_ev_t = sigma2_ev_dt[:, t]
        r2_t      = resid_sq[:, t]
        a_t       = a_omega[t]
        b_t       = b_omega[t]

        def log_post(ell: float) -> float:
            omega2_val = np.exp(ell)
            var_d      = sig2_ev_t + omega2_val
            ll = -0.5 * np.sum(np.log(var_d) + r2_t / var_d)
            # IG prior on omega^2 in ell-space (Jacobian d(omega^2)/d(ell)=omega^2 → +ell;
            # const drops out as the slice sampler uses log-density up to a constant)
            lp = -a_t * ell - b_t * np.exp(-ell)
            return ll + lp

        ell0          = np.log(max(new_omega2[t], OMEGA2_FLOOR))
        ell_new, evals = _slice_sample_1d(log_post, ell0, w=SLICE_W, rng=rng)
        total_evals  += evals
        new_omega2[t] = max(np.exp(ell_new), OMEGA2_FLOOR)

    return new_omega2, total_evals


def _slice_sample_1d(log_post, x0: float, *, w: float, rng) -> tuple[float, int]:
    """Univariate slice sampler (Neal 2003) with stepping-out + shrinkage.

    Returns (new_sample, n_log_post_evals).
    """
    n_evals = 0
    log_y = log_post(x0); n_evals += 1
    # Vertical slice: log y = log_post(x0) + log U
    log_y += np.log(rng.random() + 1e-300)

    # Initial bracket [L, R] of width w straddling x0
    u = rng.random()
    L = x0 - w * u
    R = L + w

    # Stepping out
    for _ in range(SLICE_MAX_STEPS):
        if log_post(L) <= log_y:
            n_evals += 1
            break
        n_evals += 1
        L -= w
    for _ in range(SLICE_MAX_STEPS):
        if log_post(R) <= log_y:
            n_evals += 1
            break
        n_evals += 1
        R += w

    # Shrinkage
    for _ in range(SLICE_MAX_SHRINK):
        x1 = L + (R - L) * rng.random()
        log_p_x1 = log_post(x1); n_evals += 1
        if log_p_x1 > log_y:
            return float(x1), n_evals
        if x1 < x0:
            L = x1
        else:
            R = x1

    # Shrinkage cap hit (shouldn't happen for well-behaved unimodal targets);
    # return last x1 to avoid stalling.
    return float(x1), n_evals
