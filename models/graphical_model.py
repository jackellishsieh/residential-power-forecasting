"""
Generative graphical model for residential power: fit + Gibbs inference.

Notation and derivations follow specs/model.md. Briefly:

  Per-home/day/timestep emission:

      x^(n)_{d,t} | z^(n)_{d,t}=k  ~  N( Theta^(n)_k + eta^(n)_t,
                                          (sigma^EV_k)^2 + omega^2_t(n,...) )

  with a hierarchical prior across homes on the per-home Non-EV mean
  profile eta^(n) ∈ R^T:

      eta^(n)              ~ N( eta_bar, Sigma_eta = W W^T + diag(psi) )   (PPCA / FA)

  and *one of two* parameterizations for the Non-EV variance profile,
  controlled by `omega_mode` in ModelParams:

      omega_mode = "global"        (DEFAULT, recommended for stability)
          omega^2_t = sigma2_nev_global[t]  is a GLOBAL T-vector fit from
          training data and held FIXED at inference. No Gibbs block.

      omega_mode = "hierarchical"
          (omega^(n)_t)^2 ~ InvGamma(a_omega_t, b_omega_t)   per t, per home.
          Sampled at inference via a univariate slice sampler in log-variance.

  At fit time, x_EV and x_Non-EV are observed separately. Per-home
  (eta_hat, omega_hat) are read off as empirical day-mean and within-home
  variance, then used to fit the cross-home hyperparameters.

  At inference, the latent decomposition x = x_EV + x_Non-EV is *never*
  sampled — the Gibbs sampler operates on the marginal combined-variance
  likelihood (see specs/model.md §2.5). Blocks per iter:

      1. z  via FFBS                       (HMM forward filter + backward sample)
      2. Theta_k for k in {low, high}      (conjugate Gaussian; heteroscedastic in z,t)
      3. eta                               (T-dim conjugate Gaussian under PPCA prior)
      4. omega_t (only if omega_mode == "hierarchical"; otherwise omitted)

  Plus the mixture-Gibbs C step (z = FFBS candidate vs z = all-off) and the
  heuristic-logistic resample of C from per-day z-transition rate.

The deprecated rank-1 background submodel (alpha, rho, mu_alpha, sigma_alpha,
sigma_nonev) has been removed. See specs/model.md §2.7 for what it was.

Swapping in different Non-EV parameterizations:

  All dispatch happens in three places, marked with `# DISPATCH:` comments:
    - `_fit_background()`           — fit-time dispatch on omega_mode
    - `infer_home()` initialization — choose initial omega^2
    - `infer_home()` main loop      — whether to run the omega Gibbs block
  Adding a new omega parameterization means handling these three places and
  adding a new branch in the ModelParams summary. The mean side currently
  has only one parameterization (hierarchical PPCA on eta); same pattern
  would apply if we wanted a swap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T = 24 * 60 // 15        # 15-min intervals per day  →  96
K = 3                    # number of states: 0=off, 1=low, 2=high
STATE_NAMES = ["off", "low", "high"]

# HMM / EV-side
LAPLACE = 1e-3                 # smoothing for HMM transition counts
SIGMA_EV_OFF = 1e-3            # floor for off-state EV emission std
EM_TOL = 1e-6
EM_MAX_ITERS = 100
THETA_VAR_FLOOR = 1e-6

# New Non-EV-side
PPCA_RANK_DEFAULT = 5          # rank r for Sigma_eta = W W^T + diag(psi)
PSI_FLOOR = 1e-6               # floor for per-t residual variance of eta prior
OMEGA2_FLOOR = 1e-6            # floor for per-home, per-t variance
SLICE_W = 1.5                  # slice sampler initial step (log-variance units)
SLICE_MAX_STEPS = 50           # safety cap on stepping-out iterations
SLICE_MAX_SHRINK = 50          # safety cap on shrinkage iterations

# Numerical guards for IG MoM (avoid degenerate a values)
IG_MIN_SHAPE = 2.01            # a > 2 required for finite IG variance


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """All globally point-estimated parameters of the model.

    Conventions:
      Theta_off, sigma2_theta_off, sigma2_ev_off are fixed (see specs/model.md
      §1.5–1.6).  All other parameters are estimated.

      Sigma_eta = W_eta W_eta.T + diag(psi_eta)  is the PPCA / factor-analyzer
      prior covariance for the per-home Non-EV mean profile eta^(n) (T-vec).

      omega_mode selects how the Non-EV variance profile is parameterized:
        "global"       : sigma2_nev_global is a fixed T-vector. No inference-time
                          Gibbs block. (DEFAULT; recommended.)
        "hierarchical" : (omega^(n)_t)^2 ~ InvGamma(a_omega_t, b_omega_t).
                          Sampled at inference via slice sampler.
    """

    # EV state
    p_C: float                  # EV prevalence
    pi_z: np.ndarray            # (K,) initial EV-state probabilities at t=0
    P_z: np.ndarray             # (K, K) row-stochastic transition matrix

    # EV charging magnitudes
    mu_theta:    np.ndarray     # (K,) per-state EV charging mean. mu_theta[0] = 0.
    sigma2_theta: np.ndarray    # (K,) per-state Theta prior variance. [0] = 0.
    sigma2_ev:   np.ndarray     # (K,) per-state EV emission variance. [0] = SIGMA_EV_OFF^2.

    # Non-EV: hierarchical prior on per-home mean profile eta^(n) ∈ R^T
    eta_bar: np.ndarray         # (T,) global mean profile
    W_eta:   np.ndarray         # (T, r) PPCA loading matrix
    psi_eta: np.ndarray         # (T,) per-t residual variance

    # Non-EV variance: one of two parameterizations (see class docstring).
    omega_mode: str = "global"                         # "global" | "hierarchical"
    sigma2_nev_global: np.ndarray | None = None        # (T,) — used iff omega_mode == "global"
    a_omega: np.ndarray | None = None                  # (T,) — used iff omega_mode == "hierarchical"
    b_omega: np.ndarray | None = None                  # (T,) — used iff omega_mode == "hierarchical"

    K: int = K
    T: int = T

    def __post_init__(self):
        if self.omega_mode == "global":
            if self.sigma2_nev_global is None:
                raise ValueError("omega_mode='global' requires sigma2_nev_global")
        elif self.omega_mode == "hierarchical":
            if self.a_omega is None or self.b_omega is None:
                raise ValueError("omega_mode='hierarchical' requires a_omega and b_omega")
        else:
            raise ValueError(f"unknown omega_mode={self.omega_mode!r}")

    @property
    def ppca_rank(self) -> int:
        return int(self.W_eta.shape[1])

    def Sigma_eta(self) -> np.ndarray:
        """Materialize the full T×T prior covariance (for inspection only)."""
        return self.W_eta @ self.W_eta.T + np.diag(self.psi_eta)

    def expected_omega2(self) -> np.ndarray:
        """The "best single estimate" of (omega_t)^2 per t under the current
        parameterization. For omega_mode='global', returns sigma2_nev_global;
        for 'hierarchical', returns IG prior mean b/(a-1). Shape (T,)."""
        if self.omega_mode == "global":
            return self.sigma2_nev_global
        return self.b_omega / np.maximum(self.a_omega - 1.0, 1e-12)

    def summary(self) -> str:
        r = self.ppca_rank
        lines = [
            "ModelParams summary",
            "-" * 40,
            "EV States",
            f"  p_C                 = {self.p_C:.4f}",
            f"  pi_z                = {np.array2string(self.pi_z, precision=4)}",
            f"  P_z (rows sum to 1):",
        ]
        for k in range(K):
            lines.append(f"     {STATE_NAMES[k]:>4}: {np.array2string(self.P_z[k], precision=4)}")

        lines.append("\nEV Charging Magnitudes")
        for k, name in enumerate(STATE_NAMES):
            lines.append(
                f"  Theta[{name:>4}]: mu={self.mu_theta[k]:+.4f}, "
                f"sigma_Theta={np.sqrt(self.sigma2_theta[k]):.4f}, "
                f"sigma^EV={np.sqrt(self.sigma2_ev[k]):.4f}"
            )

        lines.append("\nNon-EV — hierarchical eta prior")
        lines += [
            f"  eta_bar             (min={self.eta_bar.min():+.3f}, "
            f"median={np.median(self.eta_bar):+.3f}, "
            f"max={self.eta_bar.max():+.3f}, mean={self.eta_bar.mean():+.3f})",
            f"  W_eta               shape=(T={self.T}, r={r})",
            f"  psi_eta             (per-t residual variance: "
            f"min={self.psi_eta.min():.4f}, "
            f"median={np.median(self.psi_eta):.4f}, "
            f"max={self.psi_eta.max():.4f})",
        ]

        lines.append(f"\nNon-EV — omega parameterization: {self.omega_mode!r}")
        if self.omega_mode == "global":
            sig_g = np.sqrt(self.sigma2_nev_global)
            lines += [
                f"  sigma2_nev_global   (fixed at inference; per-t std-dev: "
                f"min={sig_g.min():.3f}, median={np.median(sig_g):.3f}, "
                f"max={sig_g.max():.3f})",
            ]
        else:
            prior_mean = self.b_omega / np.maximum(self.a_omega - 1.0, 1e-12)
            lines += [
                f"  a_omega             (IG shape: min={self.a_omega.min():.2f}, "
                f"median={np.median(self.a_omega):.2f}, "
                f"max={self.a_omega.max():.2f})",
                f"  b_omega             (IG rate:  min={self.b_omega.min():.4f}, "
                f"median={np.median(self.b_omega):.4f}, "
                f"max={self.b_omega.max():.4f})",
                f"  E[(omega_t)^2]      (prior mean: min={prior_mean.min():.4f}, "
                f"median={np.median(prior_mean):.4f}, "
                f"max={prior_mean.max():.4f})",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference container
# ---------------------------------------------------------------------------

@dataclass
class HomeInference:
    home_id: int
    C_hat: int
    z_hat: np.ndarray                            # (D, T) MAP states

    # Post-burn-in posterior summaries
    z_marginals: np.ndarray | None = None        # (D, T, K)
    eta_samples: np.ndarray | None = None        # (S, T)
    omega2_samples: np.ndarray | None = None     # (S, T)   variance, not std
    theta_samples: np.ndarray | None = None      # (S, K)

    # Per-sample C draws & helpers
    c_samples: np.ndarray | None = None                       # (S,)  int {0,1}
    c_from_z_samples: np.ndarray | None = None                # (S,)  any-nonoff indicator
    z_transitions_per_day_samples: np.ndarray | None = None   # (S,)  float

    # Full iteration traces (burn-in + retained), for convergence diagnostics
    eta_trace:      np.ndarray | None = None     # (S_burn+S, T)
    omega2_trace:   np.ndarray | None = None     # (S_burn+S, T)
    theta_trace:    np.ndarray | None = None     # (S_burn+S, K)
    state_occ_trace:np.ndarray | None = None     # (S_burn+S, K)
    loglik_trace:   np.ndarray | None = None     # (S_burn+S,)

    S_burn: int = 0


# ===========================================================================
# FIT
# ===========================================================================

def fit(
    train_df: pd.DataFrame,
    *,
    ppca_rank: int = PPCA_RANK_DEFAULT,
    omega_mode: str = "global",
    verbose: bool = True,
) -> ModelParams:
    """Fit all global parameters from a fully-labeled training dataframe.

    Required columns: home_id, day, time_index, total_load, ev_load,
                      non_ev_load, charge_state, has_ev.

    ppca_rank  : rank r for the PPCA prior covariance of eta^(n).
                 r=0 corresponds to a plain diagonal prior diag(psi).
    omega_mode : Non-EV variance parameterization.
                 "global"       — fit a single T-vector sigma2_nev_global across
                                  homes; FIXED at inference. (DEFAULT.)
                 "hierarchical" — per-home (omega^(n)_t)^2 with IG prior;
                                  sampled at inference. More flexible but can
                                  trigger a "ω shrinks → z over-fires" feedback
                                  loop in some regimes.
    """
    if verbose:
        print("=" * 60)
        print("FIT: graphical model (hierarchical Non-EV submodel)")
        print("=" * 60)

    sorted_df = train_df.sort_values(["home_id", "day", "time_index"])
    homes = sorted_df["home_id"].unique()
    N = len(homes)
    if verbose:
        print(f"\nDataset: {N} homes, {len(sorted_df):,} rows "
              f"({len(sorted_df) // T:,} home-days)")
        ev_count = sum(1 for hid in homes
                       if sorted_df.loc[sorted_df["home_id"] == hid, "has_ev"].iloc[0])
        print(f"  EV homes: {ev_count} / {N}")

    # ------------------------------------------------------------------
    # Step 1 — EV prevalence p_C
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print("\n[Step 1] EV prevalence p_C")
    p_C = sorted_df.groupby("home_id")["has_ev"].first().mean()
    if verbose:
        print(f"  p_C = {p_C:.4f} ({int(p_C * N)}/{N} homes have EV)")
        print(f"  Step 1 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Pre-shape per-home arrays for downstream steps
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Pre-shape] Building per-home (D, T) arrays")
    t0 = time.time()
    home_arrays = _build_home_arrays(sorted_df, homes)
    if verbose:
        ds = [a["D"] for a in home_arrays.values()]
        print(f"  D^(n) range: min={min(ds)}, median={int(np.median(ds))}, max={max(ds)}")
        print(f"  Pre-shape done in {time.time() - t0:.3f}s")

    ev_homes = [hid for hid in homes if home_arrays[hid]["has_ev"]]
    N_EV = len(ev_homes)

    # ------------------------------------------------------------------
    # Step 2 — HMM (EV homes only)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 2] HMM parameters from {N_EV} EV homes")
    pi_z, P_z = _fit_hmm(home_arrays, ev_homes, verbose=verbose)
    if verbose:
        print(f"  Step 2 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 3 — Hierarchical Non-EV submodel (all homes)
    #
    # 3a. eta_hat^(n)_t, omega_hat^(n)_t = empirical day-mean and per-t var
    # 3b. eta-prior: eta_bar = mean(eta_hat); PPCA(centered eta_hat) -> W, psi
    # 3c. omega-prior: MoM on (omega_hat)^2 across homes -> a_omega, b_omega
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 3] Non-EV submodel from all {N} homes  "
              f"(PPCA rank r={ppca_rank}, omega_mode={omega_mode!r})")
    eta_bar, W_eta, psi_eta, sigma2_nev_global, a_omega, b_omega = _fit_background(
        home_arrays, list(homes),
        ppca_rank=ppca_rank, omega_mode=omega_mode, verbose=verbose,
    )
    if verbose:
        print(f"  Step 3 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 4 — EV charging magnitudes (EM, EV homes only)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 4] EV charging magnitudes (EM) from {N_EV} EV homes")
    mu_theta, sigma2_theta, sigma2_ev = _fit_charging_em(home_arrays, ev_homes, verbose=verbose)
    if verbose:
        print(f"  Step 4 done in {time.time() - t0:.3f}s")

    params = ModelParams(
        p_C=float(p_C),
        pi_z=pi_z,
        P_z=P_z,
        eta_bar=eta_bar,
        W_eta=W_eta,
        psi_eta=psi_eta,
        omega_mode=omega_mode,
        sigma2_nev_global=sigma2_nev_global,
        a_omega=a_omega,
        b_omega=b_omega,
        mu_theta=mu_theta,
        sigma2_theta=sigma2_theta,
        sigma2_ev=sigma2_ev,
    )

    if verbose:
        print("\n" + params.summary())

    return params


# ---------------------------------------------------------------------------
# Helpers: per-home array assembly
# ---------------------------------------------------------------------------

def _build_home_arrays(sorted_df: pd.DataFrame, homes: Iterable[int]) -> dict:
    """Reshape long dataframe to per-home (D, T) arrays for total/ev/non-ev/state."""
    out = {}
    for hid, g in sorted_df.groupby("home_id", sort=False):
        days = g["day"].to_numpy()
        D = len(np.unique(days))
        if len(g) != D * T:
            raise ValueError(
                f"home {hid}: expected D*T={D*T} rows, got {len(g)} (incomplete days?)"
            )
        out[int(hid)] = {
            "has_ev": bool(g["has_ev"].iloc[0]),
            "city":   g["city"].iloc[0],
            "D":      D,
            "x":      g["total_load"].to_numpy().reshape(D, T).astype(np.float64),
            "x_ev":   g["ev_load"].to_numpy().reshape(D, T).astype(np.float64),
            "x_nev":  g["non_ev_load"].to_numpy().reshape(D, T).astype(np.float64),
            "z":      g["charge_state"].to_numpy().reshape(D, T).astype(np.int64),
        }
    return out


# ---------------------------------------------------------------------------
# Step 2 — HMM (unchanged from rank-1 model)
# ---------------------------------------------------------------------------

def _fit_hmm(home_arrays: dict, ev_homes: list[int], *, verbose: bool):
    pi_counts = np.zeros(K, dtype=np.float64)
    trans_counts = np.zeros((K, K), dtype=np.float64)

    for hid in ev_homes:
        z = home_arrays[hid]["z"]
        starts = z[:, 0]
        for k in range(K):
            pi_counts[k] += int(np.sum(starts == k))
        prev = z[:, :-1].ravel()
        nxt = z[:, 1:].ravel()
        for k in range(K):
            mask = (prev == k)
            if not mask.any():
                continue
            for kp in range(K):
                trans_counts[k, kp] += int(np.sum(nxt[mask] == kp))

    pi_z = pi_counts / pi_counts.sum()
    smoothed = trans_counts + LAPLACE
    P_z = smoothed / smoothed.sum(axis=1, keepdims=True)

    if verbose:
        print(f"  raw initial counts: {pi_counts.astype(int).tolist()}")
        print(f"  pi_z = {np.array2string(pi_z, precision=4)}")
        print(f"  raw transition counts (rows from→cols to):")
        for k in range(K):
            print(f"    {STATE_NAMES[k]:>4}: {trans_counts[k].astype(int).tolist()}")
        print(f"  P_z (smoothed, row-normalized):")
        for k in range(K):
            print(f"    {STATE_NAMES[k]:>4}: {np.array2string(P_z[k], precision=4)}")

    return pi_z, P_z


# ---------------------------------------------------------------------------
# Step 3 — Hierarchical Non-EV submodel (NEW)
# ---------------------------------------------------------------------------

def _fit_background(
    home_arrays: dict,
    homes: list[int],
    *,
    ppca_rank: int,
    omega_mode: str,
    verbose: bool,
):
    """Fit the Non-EV submodel from labeled training data.

    DISPATCH: omega_mode selects the variance parameterization
        "global"        → returns (sigma2_nev_global, a_omega=None, b_omega=None)
        "hierarchical"  → returns (sigma2_nev_global=None, a_omega, b_omega)

    Mean side (eta) is hierarchical PPCA in both cases.

    Returns
    -------
    eta_bar           : (T,)   global mean profile
    W_eta             : (T, r) PPCA loading matrix
    psi_eta           : (T,)   per-t residual variance
    sigma2_nev_global : (T,) or None
    a_omega           : (T,) or None
    b_omega           : (T,) or None
    """
    # --- 3a. per-home plug-in eta_hat and (omega_hat)^2 ---------------
    eta_hat   = np.stack([home_arrays[hid]["x_nev"].mean(axis=0) for hid in homes])
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

    # --- 3b. eta-prior fit (hierarchical PPCA — always) ---------------
    eta_bar, W_eta, psi_eta = _fit_eta_prior(
        eta_hat, omega2_hat, Ds, ppca_rank=ppca_rank, verbose=verbose,
    )

    # --- 3c. omega parameterization (DISPATCH on omega_mode) ----------
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


def _fit_omega_global(
    omega2_hat: np.ndarray,    # (N, T)  per-home empirical day-variance
    Ds: np.ndarray,            # (N,)    per-home day counts
    *,
    verbose: bool,
) -> np.ndarray:
    """Global per-t Non-EV variance, weighted by per-home day counts.

    sigma2_nev_global[t] = (sum_n D^(n) * omega_hat^(n)_t^2) / (sum_n D^(n))

    This is the same pooled estimator that the deprecated rank-1 model used
    (specs/model.md §2.7.4), with eta_hat^(n) as the per-home mean estimator
    instead of alpha^(n)*rho_t. Held FIXED at inference (no Gibbs block).
    """
    weights = Ds / Ds.sum()                                # (N,)
    sigma2 = (weights[:, None] * omega2_hat).sum(axis=0)   # (T,)

    if verbose:
        sig = np.sqrt(sigma2)
        print(f"  Omega-fit (global, fixed at inference):")
        print(f"    sigma_nev_global_t (std-dev): "
              f"min={sig.min():.3f}, median={np.median(sig):.3f}, "
              f"max={sig.max():.3f}")
    return sigma2


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

    NOTE: this is a *heuristic* low-rank-plus-diagonal decomposition, not
    the MLE for factor analysis (which would need EM). At N=50 and the
    relatively clean labeled training data, it's a very close approximation.
    """
    N, T_ = eta_hat.shape
    eta_bar = eta_hat.mean(axis=0)                       # (T,)
    centered = eta_hat - eta_bar                         # (N, T)

    # Sample covariance (unbiased)
    S_emp = (centered.T @ centered) / max(N - 1, 1)      # (T, T)

    # Bias subtract: within-home noise of eta_hat is omega^2 / D, averaged
    bias_diag = np.mean(omega2_hat / Ds[:, None], axis=0)   # (T,)
    S_corr = S_emp.copy()
    np.fill_diagonal(S_corr, np.diag(S_corr) - bias_diag)

    # Symmetrize (against numerical asymmetry)
    S_corr = 0.5 * (S_corr + S_corr.T)

    # Eigendecomp (eigh returns ascending; we want descending)
    eigvals, eigvecs = np.linalg.eigh(S_corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    r = int(ppca_rank)
    if r > 0:
        top_eigvals = np.maximum(eigvals[:r], 0.0)
        W_eta = eigvecs[:, :r] * np.sqrt(top_eigvals)[None, :]   # (T, r)
        # Residual after subtracting the low-rank piece; absorb diagonal into psi
        residual = S_corr - W_eta @ W_eta.T
        psi_eta = np.maximum(np.diag(residual), PSI_FLOOR)
    else:
        W_eta = np.zeros((T_, 0), dtype=np.float64)
        psi_eta = np.maximum(np.diag(S_corr), PSI_FLOOR)

    if verbose:
        # Variance explained by top-r factors (relative to bias-corrected total)
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


def _fit_omega_prior(
    omega2_hat: np.ndarray,     # (N, T)
    *,
    verbose: bool,
):
    """omega-prior fit via method-of-moments per t (InvGamma).

    Match the empirical mean and variance of {omega_hat^(n)_t)^2}_n to the
    IG mean b/(a-1) and variance b^2 / ((a-1)^2 (a-2)). Requires a > 2.

      a = m^2 / v + 2
      b = m * (a - 1)

    Floors a at IG_MIN_SHAPE.
    """
    m_omega2 = omega2_hat.mean(axis=0)                # (T,) sample mean
    v_omega2 = omega2_hat.var(axis=0, ddof=1)         # (T,) sample variance

    # Method-of-moments with floor for stability
    a_omega = np.maximum(m_omega2 ** 2 / np.maximum(v_omega2, 1e-12) + 2.0,
                          IG_MIN_SHAPE)
    b_omega = m_omega2 * (a_omega - 1.0)              # (T,)

    if verbose:
        prior_mean = b_omega / (a_omega - 1.0)
        prior_std_ratio = np.sqrt(1.0 / np.maximum(a_omega - 2.0, 1e-12))  # CV
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


# ---------------------------------------------------------------------------
# Step 4 — EV charging magnitudes (EM) — unchanged from rank-1 model
# ---------------------------------------------------------------------------

def _fit_charging_em(home_arrays: dict, ev_homes: list[int], *, verbose: bool):
    """One-way Gaussian random-effects EM for Theta_k, sigma_Theta_k, sigma^EV_k.

    Off-state (k=0) is fixed: mu_theta=0, sigma2_theta=0, sigma2_ev=SIGMA_EV_OFF^2.
    """
    mu_theta = np.zeros(K, dtype=np.float64)
    sigma2_theta = np.zeros(K, dtype=np.float64)
    sigma2_ev = np.zeros(K, dtype=np.float64)
    sigma2_ev[0] = SIGMA_EV_OFF ** 2

    for k in (1, 2):
        n_per_home, S_y, SS_y = [], [], []
        for hid in ev_homes:
            z = home_arrays[hid]["z"]
            x_ev = home_arrays[hid]["x_ev"]
            mask = (z == k)
            n_per_home.append(int(mask.sum()))
            obs = x_ev[mask]
            S_y.append(float(obs.sum()))
            SS_y.append(float((obs ** 2).sum()))
        n_per_home = np.array(n_per_home, dtype=np.float64)
        S_y = np.array(S_y, dtype=np.float64)
        SS_y = np.array(SS_y, dtype=np.float64)
        N_EV = len(ev_homes)

        if verbose:
            print(f"\n  --- State k={k} ({STATE_NAMES[k]}) ---")
            print(f"  n^(n)_k per home: min={int(n_per_home.min())}, "
                  f"median={int(np.median(n_per_home))}, max={int(n_per_home.max())}, "
                  f"sum={int(n_per_home.sum())}")
            if n_per_home.min() == 0:
                zero_homes = [ev_homes[i] for i, n in enumerate(n_per_home) if n == 0]
                print(f"  WARNING: {len(zero_homes)} home(s) never visit state {k}: {zero_homes}")

        if n_per_home.sum() == 0:
            print(f"  No observations in state {k}; skipping (using prior).")
            mu_theta[k] = 0.0
            sigma2_theta[k] = 0.0
            sigma2_ev[k] = 0.01
            continue

        active = n_per_home > 0
        theta_hat = np.where(active, S_y / np.maximum(n_per_home, 1), 0.0)

        N_k = n_per_home.sum()
        SS_within = SS_y[active].sum() - (n_per_home[active] * theta_hat[active] ** 2).sum()
        denom = N_k - active.sum()
        sigma2_ev_k = SS_within / max(denom, 1.0)
        var_theta_hat = float(np.var(theta_hat[active], ddof=1)) if active.sum() > 1 else 0.0
        mean_inv_n = float(np.mean(1.0 / n_per_home[active]))
        sigma2_theta_k = max(0.0, var_theta_hat - sigma2_ev_k * mean_inv_n)
        mu_theta_k = float(theta_hat[active].mean())

        if verbose:
            print(f"  ANOVA init: mu={mu_theta_k:.4f}, "
                  f"sigma_Theta={np.sqrt(sigma2_theta_k):.4f}, "
                  f"sigma^EV={np.sqrt(sigma2_ev_k):.4f}")

        prev_loglik = -np.inf
        for it in range(EM_MAX_ITERS):
            sig2_theta = max(sigma2_theta_k, THETA_VAR_FLOOR)

            prec = 1.0 / sig2_theta + n_per_home / sigma2_ev_k
            E_theta = (mu_theta_k / sig2_theta + S_y / sigma2_ev_k) / prec
            Var_theta = 1.0 / prec
            E_theta2 = Var_theta + E_theta ** 2

            loglik = _charging_loglik(n_per_home, theta_hat, S_y, SS_y,
                                       mu_theta_k, sig2_theta, sigma2_ev_k, active)

            mu_theta_k = float(np.mean(E_theta))
            sigma2_theta_k = float(np.mean(Var_theta + (E_theta - mu_theta_k) ** 2))
            sigma2_ev_k = float(
                np.sum(SS_y - 2 * S_y * E_theta + n_per_home * E_theta2) / N_k
            )

            delta = loglik - prev_loglik
            if verbose and (it < 5 or it % 10 == 0 or abs(delta) < EM_TOL):
                print(f"    iter {it:3d}: logL={loglik:.4f}  Δ={delta:+.2e}  "
                      f"mu={mu_theta_k:.4f}  σΘ={np.sqrt(max(sigma2_theta_k,0)):.4f}  "
                      f"σEV={np.sqrt(sigma2_ev_k):.4f}")

            if abs(delta) < EM_TOL and it > 0:
                if verbose:
                    print(f"  EM converged at iter {it}")
                break
            prev_loglik = loglik

        mu_theta[k] = mu_theta_k
        sigma2_theta[k] = max(sigma2_theta_k, 0.0)
        sigma2_ev[k] = sigma2_ev_k

    return mu_theta, sigma2_theta, sigma2_ev


def _charging_loglik(n, theta_hat, S_y, SS_y, mu, sig2_theta, sig2_ev, active):
    """Marginal log-likelihood of one-way Gaussian RE model (sum over homes)."""
    n_a = n[active]
    th_a = theta_hat[active]
    SS_a = SS_y[active]
    within_ss = SS_a - n_a * th_a ** 2

    ll = -0.5 * np.sum(
        n_a * np.log(2 * np.pi)
        + (n_a - 1) * np.log(sig2_ev)
        + np.log(sig2_ev + n_a * sig2_theta)
        + within_ss / sig2_ev
        + n_a * (th_a - mu) ** 2 / (sig2_ev + n_a * sig2_theta)
    )
    return float(ll)


# ===========================================================================
# INFERENCE
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
    record_traces: bool = True,
    initial_c: int = 1,
    initial_z: np.ndarray | None = None,
    c_logistic_model=None,
) -> HomeInference:
    """Mixture Gibbs for one home, with the new hierarchical Non-EV submodel.

    home_x           : (D, T) total grid power — the only signal at test time.
    initial_c        : warm-start C value (0 or 1).
    initial_z        : (D, T) warm-start z; defaults to all-off if None.
    c_logistic_model : fitted sklearn LogisticRegression on transitions/day.
                       If None, falls back to a hard threshold at 1.0.

    Per-iteration block structure (see specs/model.md §4.1):

      A. Mixture-Gibbs z step (HMM marginal):
           A1. FFBS proposes z_candidate, returns log_Z1 (HMM marginal)
           A2. compute log_Z0 (z=all-off marginal)
           A3. set z = z_candidate w.p. softmax(p_C·exp(log_Z1), (1-p_C)·exp(log_Z0))
       B. C | z step via logistic regression on transitions/day
       1. (already done in A) FFBS for z
       2. Theta_k for k in {low, high}    — conjugate Gaussian
       3. eta                              — T-dim conjugate Gaussian under PPCA prior
       4. (omega_t^2) for t = 0..T-1       — univariate slice sampler in log-variance

    The latent decomposition x = x_EV + x_Non-EV stays marginalized
    throughout — see specs/model.md §2.5 for the rationale.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"

    if verbose:
        print(f"  [home {home_id}] D={D} → "
              f"hierarchical mixture Gibbs ({S_burn} burn-in + {S} retained)  "
              f"initial_c={initial_c}")

    # ── initial state ─────────────────────────────────────────────────────────
    theta  = params.mu_theta.copy()
    eta    = params.eta_bar.copy()

    # DISPATCH: initial omega^2 depends on the variance parameterization
    if params.omega_mode == "global":
        # Fixed across all iterations; the omega Gibbs block is omitted below
        omega2 = params.sigma2_nev_global.copy()
    elif params.omega_mode == "hierarchical":
        # IG mode (slightly more conservative than the mean when a is small)
        omega2 = (params.b_omega / (params.a_omega + 1.0)).copy()
    else:
        raise ValueError(f"unknown omega_mode={params.omega_mode!r}")

    z      = initial_z.copy() if initial_z is not None else np.zeros((D, T), dtype=np.int64)
    c      = initial_c

    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    # Precompute Sigma_eta^{-1} once per omega update; for now compute once at init.
    Sigma_eta_inv = _compute_sigma_eta_inv(params.W_eta, params.psi_eta)
    Sigma_eta_inv_etabar = Sigma_eta_inv @ params.eta_bar    # (T,)

    # ── storage ───────────────────────────────────────────────────────────────
    n_total = S_burn + S
    z_counts                      = np.zeros((D, T, K), dtype=np.float64)
    eta_samples                   = np.zeros((S, T),    dtype=np.float64)
    omega2_samples                = np.zeros((S, T),    dtype=np.float64)
    theta_samples                 = np.zeros((S, K),    dtype=np.float64)
    c_samples                     = np.zeros(S,         dtype=np.int8)
    c_from_z_samples              = np.zeros(S,         dtype=np.int8)
    z_transitions_per_day_samples = np.zeros(S,         dtype=np.float64)

    if record_traces:
        eta_trace       = np.zeros((n_total, T), dtype=np.float64)
        omega2_trace    = np.zeros((n_total, T), dtype=np.float64)
        theta_trace     = np.zeros((n_total, K), dtype=np.float64)
        state_occ_trace = np.zeros((n_total, K), dtype=np.float64)
        loglik_trace    = np.zeros(n_total,      dtype=np.float64)
    else:
        eta_trace = omega2_trace = theta_trace = state_occ_trace = loglik_trace = None

    # ── main loop ─────────────────────────────────────────────────────────────
    t_start = time.time()
    s_idx = -1
    n_slice_evals = 0  # track slice-sampler cost

    for it in range(n_total):

        # --- A. mixture-Gibbs z step ----------------------------------------
        z_candidate, log_Z1 = _ffbs(home_x, theta, eta, omega2, params, log_pi, log_P, rng)
        log_Z0 = _compute_loglik_c0(home_x, eta, omega2, params)
        log_w1 = np.log(params.p_C + 1e-300)     + log_Z1
        log_w0 = np.log(1 - params.p_C + 1e-300) + log_Z0
        p_c1_eff = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))
        z = z_candidate if rng.random() < np.clip(p_c1_eff, 0.0, 1.0) \
            else np.zeros((D, T), dtype=np.int64)

        # --- B. C | z step (transitions/day → logistic) ---------------------
        transitions_per_day_now = float((np.diff(z, axis=1) != 0).sum() / D)
        if c_logistic_model is not None:
            p_c1 = float(c_logistic_model.predict_proba([[transitions_per_day_now]])[0, 1])
        else:
            p_c1 = float(transitions_per_day_now > 1.0)
        c = int(rng.random() < p_c1)

        # --- 2. Theta_k -----------------------------------------------------
        for k in (1, 2):
            theta[k] = _sample_theta_k(home_x, z, eta, omega2, params, k, rng)

        # --- 3. eta (T-dim conjugate Gaussian under PPCA prior) -------------
        eta = _sample_eta(home_x, z, theta, omega2, params,
                          Sigma_eta_inv, Sigma_eta_inv_etabar, rng)

        # --- 4. omega^2 ----------------------------------------------------
        # DISPATCH: only sample omega when the parameterization makes it a
        # latent. In "global" mode, omega2 is fixed at sigma2_nev_global.
        if params.omega_mode == "hierarchical":
            omega2, evals_this_iter = _sample_omega(
                home_x, z, theta, eta, omega2, params, rng,
            )
            n_slice_evals += evals_this_iter

        # --- traces ---------------------------------------------------------
        if record_traces:
            eta_trace[it]       = eta
            omega2_trace[it]    = omega2
            theta_trace[it]     = theta
            state_occ_trace[it] = [(z == k).mean() for k in range(K)]
            loglik_trace[it]    = _compute_loglik(home_x, z, theta, eta, omega2, params)

        # --- accumulate post-burn-in -----------------------------------------
        if it >= S_burn:
            s_idx = it - S_burn
            eta_samples[s_idx]    = eta
            omega2_samples[s_idx] = omega2
            theta_samples[s_idx]  = theta
            c_samples[s_idx]      = c
            for k in range(K):
                z_counts[:, :, k] += (z == k)
            c_from_z_samples[s_idx]              = int(np.any(z != 0))
            z_transitions_per_day_samples[s_idx] = float(
                (np.diff(z, axis=1) != 0).sum() / D
            )

        # --- progress -------------------------------------------------------
        if verbose and (it < 3 or it == S_burn or (it + 1) % 100 == 0):
            phase = "burn-in" if it < S_burn else "keep  "
            elapsed = time.time() - t_start
            ll = loglik_trace[it] if record_traces else float("nan")
            slice_tag = (
                f", slice~{n_slice_evals / max(it+1,1):.1f}eval/it"
                if params.omega_mode == "hierarchical" else ""
            )
            print(f"    iter {it+1:4d}/{n_total} [{phase}]  "
                  f"C={c}  Θ_low={theta[1]:.3f}  Θ_high={theta[2]:.3f}  "
                  f"η∈[{eta.min():+.2f},{eta.max():+.2f}]  "
                  f"σω∈[{np.sqrt(omega2.min()):.3f},{np.sqrt(omega2.max()):.3f}]  "
                  f"logL={ll:.1f}  ({elapsed:.1f}s{slice_tag})")

    # ── final summaries ───────────────────────────────────────────────────────
    z_marginals = z_counts / S
    z_hat       = np.argmax(z_marginals, axis=2)
    c_hat_prob  = float(c_samples.mean())

    if verbose:
        elapsed = time.time() - t_start
        frac = z_marginals.mean(axis=(0, 1))
        slice_tag = (
            f"  (total slice evals = {n_slice_evals}, "
            f"avg {n_slice_evals/n_total:.1f}/iter across T={T})"
            if params.omega_mode == "hierarchical" else ""
        )
        print(f"\n  [home {home_id}] done in {elapsed:.1f}s{slice_tag}")
        print(f"    P̂(C=1) from chain : {c_hat_prob:.4f}  (hard={int(c_hat_prob >= 0.5)})")
        print(f"    z freq : off={frac[0]:.3f}  low={frac[1]:.3f}  high={frac[2]:.3f}")
        eta_post_mean    = eta_samples.mean(axis=0)
        omega2_post_mean = omega2_samples.mean(axis=0)
        print(f"    eta posterior mean: min={eta_post_mean.min():+.3f} "
              f"median={np.median(eta_post_mean):+.3f} max={eta_post_mean.max():+.3f}")
        print(f"    omega^2 posterior mean: min={omega2_post_mean.min():.4f} "
              f"median={np.median(omega2_post_mean):.4f} max={omega2_post_mean.max():.4f}")
        for k in (1, 2):
            print(f"    Θ[{STATE_NAMES[k]:>4}] : "
                  f"mean={theta_samples[:,k].mean():.3f}  std={theta_samples[:,k].std():.4f}")

    return HomeInference(
        home_id                       = home_id,
        C_hat                         = int(c_hat_prob >= 0.5),
        z_hat                         = z_hat,
        z_marginals                   = z_marginals,
        eta_samples                   = eta_samples,
        omega2_samples                = omega2_samples,
        theta_samples                 = theta_samples,
        c_samples                     = c_samples,
        c_from_z_samples              = c_from_z_samples,
        z_transitions_per_day_samples = z_transitions_per_day_samples,
        eta_trace                     = eta_trace,
        omega2_trace                  = omega2_trace,
        theta_trace                   = theta_trace,
        state_occ_trace               = state_occ_trace,
        loglik_trace                  = loglik_trace,
        S_burn                        = S_burn,
    )


# ---------------------------------------------------------------------------
# Likelihoods
# ---------------------------------------------------------------------------

def _compute_loglik(
    x: np.ndarray,         # (D, T)
    z: np.ndarray,         # (D, T)
    theta: np.ndarray,     # (K,)
    eta:   np.ndarray,     # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
) -> float:
    """Complete-data log-likelihood under the new model:

        sum_{d,t} log N( x[d,t] ; theta[z[d,t]] + eta[t],
                                  sigma2_ev[z[d,t]] + omega2[t] )
    """
    var_dt  = params.sigma2_ev[z] + omega2[None, :]
    mean_dt = theta[z] + eta[None, :]
    ll = -0.5 * (np.log(2 * np.pi * var_dt) + (x - mean_dt) ** 2 / var_dt)
    return float(ll.sum())


def _compute_loglik_c0(
    x: np.ndarray,         # (D, T)
    eta: np.ndarray,       # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
) -> float:
    """log p(x | C=0) = log p(x | z≡off) under the new model.

        sum_{d,t} log N( x[d,t] ; 0 + eta[t], sigma2_ev[off] + omega2[t] )
    """
    var_t = params.sigma2_ev[0] + omega2                  # (T,)
    residual = x - eta[None, :]                           # (D, T)
    ll = -0.5 * (np.log(2 * np.pi * var_t[None, :])
                 + residual ** 2 / var_t[None, :])
    return float(ll.sum())


# ---------------------------------------------------------------------------
# Gibbs block 1 — FFBS (forward filter, backward sample)
# ---------------------------------------------------------------------------

def _ffbs(
    x:      np.ndarray,    # (D, T)
    theta:  np.ndarray,    # (K,)
    eta:    np.ndarray,    # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
    log_pi: np.ndarray,
    log_P:  np.ndarray,
    rng,
):
    """Vectorized FFBS over days.

    Emission per (d, t, k):
        N( x[d,t] ; theta[k] + eta[t], sigma2_ev[k] + omega2[t] )

    Returns (z, log_Z1) where log_Z1 = log p(x | C=1, params), accumulated
    as the sum of per-step log-normalization constants in the forward pass.
    """
    D = x.shape[0]
    sigma2_ev = params.sigma2_ev

    combined_var = sigma2_ev[:, None] + omega2[None, :]        # (K, T)
    inv_2var = 0.5 / combined_var
    log_norm = -0.5 * np.log(2 * np.pi * combined_var)          # (K, T)

    mean_kt = theta[:, None] + eta[None, :]                     # (K, T)
    diff = x[:, :, None] - mean_kt.T[None, :, :]                # (D, T, K)
    log_emit = log_norm.T[None, :, :] - diff ** 2 * inv_2var.T[None, :, :]  # (D, T, K)

    # forward
    log_f  = np.empty((D, T, K), dtype=np.float64)
    log_Z1 = 0.0

    unnorm_0 = log_pi[None, :] + log_emit[:, 0, :]               # (D, K)
    lse_0    = logsumexp(unnorm_0, axis=1)
    log_Z1  += lse_0.sum()
    log_f[:, 0, :] = unnorm_0 - lse_0[:, None]

    for t in range(1, T):
        log_pred = logsumexp(log_f[:, t-1, :, None] + log_P[None, :, :], axis=1)
        unnorm_t = log_emit[:, t, :] + log_pred
        lse_t    = logsumexp(unnorm_t, axis=1)
        log_Z1  += lse_t.sum()
        log_f[:, t, :] = unnorm_t - lse_t[:, None]

    # backward sample
    z = np.empty((D, T), dtype=np.int64)
    p_T = np.exp(log_f[:, T-1, :])
    p_T /= p_T.sum(axis=1, keepdims=True)
    z[:, T-1] = _sample_categorical_rows(p_T, rng)

    P_z = params.P_z
    for t in range(T - 2, -1, -1):
        col = P_z[:, z[:, t+1]].T                               # (D, K)
        w = np.exp(log_f[:, t, :]) * col
        w /= w.sum(axis=1, keepdims=True)
        z[:, t] = _sample_categorical_rows(w, rng)

    return z, log_Z1


def _sample_categorical_rows(probs: np.ndarray, rng) -> np.ndarray:
    cum = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0])[:, None]
    return np.argmax(cum > u, axis=1)


# ---------------------------------------------------------------------------
# Gibbs block 2 — Theta_k (conjugate Gaussian, marginal likelihood)
# ---------------------------------------------------------------------------

def _sample_theta_k(
    x:      np.ndarray,    # (D, T)
    z:      np.ndarray,    # (D, T)
    eta:    np.ndarray,    # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
    k:      int,
    rng,
) -> float:
    """Sample Theta_k from its conditional Gaussian under the marginal model:

        x[d,t] - eta[t]  ~  N( Theta_k, sigma2_ev[k] + omega2[t] )  for (d,t) ∈ T_k

    Heteroscedastic across (d,t) because the variance depends only on t once
    we condition on z[d,t]=k, but t varies within the masked set.
    """
    sigma2_ev_k = params.sigma2_ev[k]
    sig2_prior  = max(params.sigma2_theta[k], THETA_VAR_FLOOR)

    mask = (z == k)                                              # (D, T)
    if not mask.any():
        return rng.normal(params.mu_theta[k], np.sqrt(sig2_prior))

    var_t   = sigma2_ev_k + omega2                                # (T,)
    inv_var_t = 1.0 / var_t                                       # (T,)

    # Per-t sums of (1/var) and (residual/var), then sum over t restricted to mask
    r = x - eta[None, :]                                          # (D, T)
    S_inv_var = (mask * inv_var_t[None, :]).sum()
    S_r       = (mask * r * inv_var_t[None, :]).sum()

    prec = 1.0 / sig2_prior + S_inv_var
    m    = (params.mu_theta[k] / sig2_prior + S_r) / prec
    return rng.normal(m, np.sqrt(1.0 / prec))


# ---------------------------------------------------------------------------
# Gibbs block 3 — eta (T-dim conjugate Gaussian under PPCA prior)
# ---------------------------------------------------------------------------

def _compute_sigma_eta_inv(W: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Sigma_eta^{-1} = (W W^T + diag(psi))^{-1} via Woodbury.

        = diag(1/psi) - diag(1/psi) W (I_r + W^T diag(1/psi) W)^{-1} W^T diag(1/psi)
    """
    T_ = psi.shape[0]
    r = W.shape[1]
    inv_psi = 1.0 / psi
    if r == 0:
        return np.diag(inv_psi)
    M = np.eye(r) + W.T @ (inv_psi[:, None] * W)                  # (r, r)
    WP = inv_psi[:, None] * W                                      # (T, r)
    return np.diag(inv_psi) - WP @ np.linalg.solve(M, WP.T)


def _sample_eta(
    x:      np.ndarray,    # (D, T)
    z:      np.ndarray,    # (D, T)
    theta:  np.ndarray,    # (K,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
    Sigma_eta_inv:           np.ndarray,   # (T, T) cached
    Sigma_eta_inv_etabar:    np.ndarray,   # (T,)   cached
    rng,
) -> np.ndarray:
    """T-dim conjugate Gaussian sample for eta under PPCA prior.

    Likelihood: x[d,t] - theta[z[d,t]]  ~  N(eta[t], sigma2_ev[z[d,t]] + omega2[t])
                                          (heteroscedastic in (d,t))

    Posterior precision: Sigma_eta^{-1} + diag(lambda_t)
                         where lambda_t = sum_d 1/(sigma2_ev[z[d,t]] + omega2[t])

    Posterior mean:      Sigma_post (Sigma_eta^{-1} eta_bar + h_data)
                         where h_data[t] = sum_d (x[d,t] - theta[z[d,t]])
                                                 / (sigma2_ev[z[d,t]] + omega2[t])
    """
    D, T_ = x.shape

    var_dt     = params.sigma2_ev[z] + omega2[None, :]            # (D, T)
    inv_var_dt = 1.0 / var_dt                                      # (D, T)

    lambda_t = inv_var_dt.sum(axis=0)                              # (T,)
    h_data   = ((x - theta[z]) * inv_var_dt).sum(axis=0)           # (T,)

    # Posterior precision
    Lambda = Sigma_eta_inv.copy()
    Lambda.flat[::T_ + 1] += lambda_t   # add lambda_t to diagonal

    # Info vector h
    h = Sigma_eta_inv_etabar + h_data                              # (T,)

    # Solve Lambda mu = h, then sample eta = mu + Lambda^{-T/2} z
    L = np.linalg.cholesky(Lambda)                                 # Lambda = L L^T
    mu = np.linalg.solve(L.T, np.linalg.solve(L, h))
    # Sample: cov = Lambda^{-1} = L^{-T} L^{-1}; draw L^{-T} ξ  with ξ ~ N(0, I)
    xi = rng.standard_normal(T_)
    return mu + np.linalg.solve(L.T, xi)


# ---------------------------------------------------------------------------
# Gibbs block 4 — omega (univariate slice sampler in log-variance, per t)
# ---------------------------------------------------------------------------

def _sample_omega(
    x:      np.ndarray,    # (D, T)
    z:      np.ndarray,    # (D, T)
    theta:  np.ndarray,    # (K,)
    eta:    np.ndarray,    # (T,)
    omega2: np.ndarray,    # (T,) current state — updated in place per t
    params: ModelParams,
    rng,
):
    """Slice-sample (omega_t)^2 for each t independently, in log-variance space.

    Posterior per t (with l = log(omega^2)):

        log p(l | rest) = log p_IG(exp(l); a_omega_t, b_omega_t) + l
                       - 0.5 * Σ_d [ log(sigma2_ev[z[d,t]] + exp(l))
                                     + (x[d,t] - theta[z[d,t]] - eta[t])^2
                                       / (sigma2_ev[z[d,t]] + exp(l)) ]

    Returns (new_omega2 (T,), total_log_density_evals (int)).
    """
    D, T_ = x.shape
    new_omega2 = omega2.copy()
    total_evals = 0

    sigma2_ev_dt = params.sigma2_ev[z]   # (D, T) — used per-t below

    # Residual per (d, t): x[d,t] - theta[z[d,t]] - eta[t]
    resid = x - theta[z] - eta[None, :]   # (D, T)
    resid_sq = resid ** 2                  # (D, T)

    a_omega = params.a_omega
    b_omega = params.b_omega

    for t in range(T_):
        sig2_ev_t = sigma2_ev_dt[:, t]   # (D,)
        r2_t      = resid_sq[:, t]       # (D,)
        a_t       = a_omega[t]
        b_t       = b_omega[t]

        def log_post(ell: float) -> float:
            # ell = log(omega^2)
            omega2_val = np.exp(ell)
            var_d = sig2_ev_t + omega2_val
            # Likelihood part
            ll = -0.5 * np.sum(np.log(var_d) + r2_t / var_d)
            # Prior (in ell-space): IG density on omega^2 with Jacobian
            #   log p_IG(omega^2) = a log(b) - lgamma(a) - (a+1) log(omega^2) - b/omega^2
            #   + Jacobian d(omega^2)/d(ell) = omega^2  →  +ell
            # const drops out (slice uses log-density up to a constant)
            lp = -a_t * ell - b_t * np.exp(-ell)
            return ll + lp

        ell0 = np.log(max(new_omega2[t], OMEGA2_FLOOR))
        ell_new, evals = _slice_sample_1d(log_post, ell0, w=SLICE_W, rng=rng)
        total_evals += evals
        new_omega2[t] = max(np.exp(ell_new), OMEGA2_FLOOR)

    return new_omega2, total_evals


def _slice_sample_1d(
    log_post,
    x0: float,
    *,
    w: float,
    rng,
) -> tuple[float, int]:
    """Univariate slice sampler (Neal 2003) with stepping-out + shrinkage.

    Returns (new_sample, n_log_post_evals).
    """
    n_evals = 0
    log_y = log_post(x0); n_evals += 1
    # Vertical slice: y = uniform(0, p(x0))  ->  log y = log_post(x0) + log(U)
    log_y += np.log(rng.random() + 1e-300)

    # Initial interval [L, R] of width w straddling x0
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
    else:
        # Hit cap; bail with current L
        pass

    for _ in range(SLICE_MAX_STEPS):
        if log_post(R) <= log_y:
            n_evals += 1
            break
        n_evals += 1
        R += w
    else:
        pass

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


# ===========================================================================
# Higher-level: run inference over a whole training-style dataframe
# ===========================================================================

def infer_all(
    df: pd.DataFrame,
    params: ModelParams,
    *,
    S_burn: int = 200,
    S: int = 500,
    seed: int = 0,
    verbose: bool = True,
    initial_c_dict: dict[int, int] | None = None,
    initial_z_dict: dict[int, np.ndarray] | None = None,
    c_logistic_model=None,
) -> dict[int, HomeInference]:
    """Run mixture Gibbs on every home in df."""
    if verbose:
        print("=" * 60)
        print("INFERENCE: hierarchical mixture Gibbs over all homes")
        print("=" * 60)

    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    homes = list(sorted_df["home_id"].unique())
    rng = np.random.default_rng(seed)

    results: dict[int, HomeInference] = {}
    t0 = time.time()
    for i, hid in enumerate(homes):
        g = sorted_df[sorted_df["home_id"] == hid]
        D = len(g) // T
        x = g["total_load"].to_numpy().reshape(D, T).astype(np.float64)

        init_c = int((initial_c_dict or {}).get(int(hid), 1))
        init_z = (initial_z_dict or {}).get(int(hid), None)

        if verbose:
            true_c = int(g["has_ev"].iloc[0]) if "has_ev" in g.columns else "?"
            print(f"\n[{i+1}/{len(homes)}] home {hid}  "
                  f"D={D}  true_c={true_c}  init_c={init_c}")

        results[int(hid)] = infer_home(
            x, params,
            S_burn=S_burn, S=S,
            rng=rng, home_id=int(hid), verbose=verbose,
            initial_c=init_c, initial_z=init_z,
            c_logistic_model=c_logistic_model,
        )

    if verbose:
        print(f"\nAll homes done in {time.time() - t0:.1f}s")

    return results


# ===========================================================================
# Heuristic adapter (unchanged)
# ===========================================================================

def c_prob_from_z_via_heuristic(
    inference: "HomeInference",
    logistic_model,
) -> float:
    """P̂(C=1) from per-sample transition rates fed through a heuristic logistic."""
    rates = inference.z_transitions_per_day_samples
    probs = logistic_model.predict_proba(rates.reshape(-1, 1))[:, 1]
    return float(probs.mean())


def build_heuristic_homes(df: pd.DataFrame) -> dict:
    """Reconstruct {dataid: (has_car, city, df_with_load_col)} from flat train_df."""
    out = {}
    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    for hid, g in sorted_df.groupby("home_id", sort=False):
        out[int(hid)] = (
            bool(g["has_ev"].iloc[0]),
            g["city"].iloc[0],
            g[["total_load"]].rename(columns={"total_load": "load"}).reset_index(drop=True),
        )
    return out


# ===========================================================================
# Evaluation (unchanged)
# ===========================================================================

def evaluate(
    df: pd.DataFrame,
    inferences: dict[int, HomeInference],
    c_prob_methods: dict[str, dict[int, float]] | None = None,
    heuristic_states: dict[int, np.ndarray] | None = None,
) -> dict:
    sorted_df = df.sort_values(["home_id", "day", "time_index"])

    ev_hard_cms, ev_soft_cms, ev_heur_cms = [], [], []
    non_ev_hard_cms, non_ev_soft_cms, non_ev_heur_cms = [], [], []
    ev_home_ids, non_ev_home_ids = [], []

    for hid, g in sorted_df.groupby("home_id", sort=True):
        hid = int(hid)
        if hid not in inferences:
            continue
        C_true = int(g["has_ev"].iloc[0])
        D = g["day"].nunique()
        z_true = g["charge_state"].to_numpy().reshape(D, T)
        inf = inferences[hid]

        hard_cm = _per_home_z_confusion_hard(z_true, inf.z_hat)
        soft_cm = (
            _per_home_z_confusion_soft(z_true, inf.z_marginals)
            if inf.z_marginals is not None else None
        )
        heur_cm = None
        if heuristic_states and hid in heuristic_states:
            heur_z = heuristic_states[hid][: D * T].reshape(D, T)
            heur_cm = _per_home_z_confusion_hard(z_true, heur_z)

        if C_true == 1:
            ev_home_ids.append(hid)
            ev_hard_cms.append(hard_cm)
            if soft_cm is not None:
                ev_soft_cms.append(soft_cm)
            if heur_cm is not None:
                ev_heur_cms.append(heur_cm)
        else:
            non_ev_home_ids.append(hid)
            non_ev_hard_cms.append(hard_cm)
            if soft_cm is not None:
                non_ev_soft_cms.append(soft_cm)
            if heur_cm is not None:
                non_ev_heur_cms.append(heur_cm)

    c_results: dict[str, dict] = {}
    for method_name, c_probs in (c_prob_methods or {}).items():
        c_results[method_name] = _c_confusion_from_probs(sorted_df, inferences, c_probs)

    return {
        "ev_home_ids":   ev_home_ids,
        "ev_z_hard":     _nanmean_cms(ev_hard_cms),
        "ev_z_soft":     _nanmean_cms(ev_soft_cms) if ev_soft_cms else None,
        "ev_z_heur":     _nanmean_cms(ev_heur_cms) if ev_heur_cms else None,
        "non_ev_home_ids": non_ev_home_ids,
        "non_ev_z_hard": _nanmean_cms(non_ev_hard_cms),
        "non_ev_z_soft": _nanmean_cms(non_ev_soft_cms) if non_ev_soft_cms else None,
        "non_ev_z_heur": _nanmean_cms(non_ev_heur_cms) if non_ev_heur_cms else None,
        "c_results":     c_results,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers (unchanged)
# ---------------------------------------------------------------------------

def _per_home_z_confusion_hard(z_true, z_pred):
    cm = np.full((K, K), np.nan)
    for k_true in range(K):
        mask = (z_true == k_true)
        n = int(mask.sum())
        if n == 0:
            continue
        for k_pred in range(K):
            cm[k_true, k_pred] = float((z_pred[mask] == k_pred).sum()) / n
    return cm


def _per_home_z_confusion_soft(z_true, z_marginals):
    cm = np.full((K, K), np.nan)
    for k_true in range(K):
        mask = (z_true == k_true)
        n = int(mask.sum())
        if n == 0:
            continue
        cm[k_true] = z_marginals[mask].sum(axis=0) / n
    return cm


def _nanmean_cms(cm_list):
    if not cm_list:
        return None
    return np.nanmean(np.stack(cm_list, axis=0), axis=0)


def _c_confusion_from_probs(sorted_df, inferences, c_probs):
    rows = []
    for hid, g in sorted_df.groupby("home_id", sort=True):
        hid = int(hid)
        if hid not in inferences or hid not in c_probs:
            continue
        C_true = int(g["has_ev"].iloc[0])
        p_hat = float(c_probs[hid])
        rows.append((C_true, int(p_hat >= 0.5), p_hat))

    hard_cm = np.zeros((2, 2), dtype=float)
    soft_cm = np.zeros((2, 2), dtype=float)
    counts = np.zeros(2, dtype=int)
    for C_true, C_hard, p_hat in rows:
        hard_cm[C_true, C_hard] += 1
        soft_cm[C_true, 0] += 1 - p_hat
        soft_cm[C_true, 1] += p_hat
        counts[C_true] += 1

    with np.errstate(invalid="ignore"):
        hard_cm_norm = np.where(counts[:, None] > 0, hard_cm / counts[:, None], np.nan)
        soft_cm_norm = np.where(counts[:, None] > 0, soft_cm / counts[:, None], np.nan)

    n_correct = sum(1 for C_true, C_hard, _ in rows if C_true == C_hard)
    return {
        "hard_cm":  hard_cm_norm,
        "soft_cm":  soft_cm_norm,
        "accuracy": float(n_correct / max(len(rows), 1)),
        "n_homes":  len(rows),
        "n_ev":     int(counts[1]),
        "n_non_ev": int(counts[0]),
    }


# ---------------------------------------------------------------------------
# Printing (unchanged)
# ---------------------------------------------------------------------------

def print_evaluation(results: dict) -> None:
    SEP = "─" * 64

    def _fmt_row(label, row, n=None):
        cells = "  ".join(
            f"{'NaN':>7}" if np.isnan(v) else f"{v:>7.3f}" for v in row
        )
        suffix = f"  (n={n})" if n is not None else ""
        return f"  {label:<8} {cells}{suffix}"

    for group_label, home_ids, hard_cm, soft_cm, heur_cm in [
        ("EV homes (C_true=1)",
         results["ev_home_ids"],
         results["ev_z_hard"], results["ev_z_soft"], results["ev_z_heur"]),
        ("non-EV homes (C_true=0)",
         results["non_ev_home_ids"],
         results["non_ev_z_hard"], results["non_ev_z_soft"], results["non_ev_z_heur"]),
    ]:
        n_homes = len(home_ids)
        print(f"\n{SEP}")
        print(f"z confusion — {group_label}  (N={n_homes} homes)")
        print(f"  Aggregation: per-home row-normalised CM, then mean over {n_homes} homes")
        print(f"  Rows = true state, columns = predicted state")
        if group_label.startswith("non-EV"):
            print("  Note: rows 'low' and 'high' are NaN (no ground-truth examples)")
        header = f"  {'':8}  {'off':>7}  {'low':>7}  {'high':>7}"

        for cm, variant in [(hard_cm, "hard (MAP z)"),
                            (soft_cm, "soft (posterior)"),
                            (heur_cm, "hard (heuristic baseline)")]:
            if cm is None:
                continue
            print(f"\n  [{variant}]")
            print(header)
            for k, name in enumerate(STATE_NAMES):
                print(_fmt_row(name, cm[k]))

    for method_name, cr in results.get("c_results", {}).items():
        print(f"\n{SEP}")
        print(f"C confusion — method: {method_name}")
        print(f"  Aggregation: row-normalised CM averaged over {cr['n_homes']} homes")
        print(f"  ({cr['n_ev']} EV, {cr['n_non_ev']} non-EV)  accuracy={cr['accuracy']:.4f}")
        print(f"  Rows = true C, columns = predicted C")
        header = f"  {'':8}  {'no-EV':>7}  {'EV':>7}"
        for cm, variant in [(cr["hard_cm"], "hard (threshold 0.5)"),
                            (cr["soft_cm"], "soft (P̂ as fraction)")]:
            print(f"\n  [{variant}]")
            print(header)
            for k, name in enumerate(["no-EV", "EV"]):
                print(_fmt_row(name, cm[k]))


def _confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def format_confusion(cm, labels):
    n = cm.shape[0]
    header = "          " + "  ".join(f"{l:>8}" for l in labels) + "   total"
    lines = [header]
    for i in range(n):
        row = cm[i]
        lines.append(
            f"  {labels[i]:>6} | " + "  ".join(f"{int(c):>8}" for c in row)
            + f"   {int(row.sum()):>8}"
        )
    col_totals = cm.sum(axis=0)
    lines.append(
        "  " + " " * 6 + " | " + "  ".join(f"{int(c):>8}" for c in col_totals)
        + f"   {int(cm.sum()):>8}"
    )
    return "\n".join(lines)
