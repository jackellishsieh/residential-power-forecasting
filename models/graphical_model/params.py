"""Constants, parameter container, and inference container.

All values fit by `fit()` live in `ModelParams`; all posterior summaries
produced by `infer_home*()` live in `HomeInference`. These are the two
dataclasses that flow between training, inference, and evaluation.

Indexing conventions follow specs/model.md:
    T = 96    intraday timesteps (15-min cells in a day)
    K = 3     EV charging states  (0=off, 1=low, 2=high)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Shape / state constants
# ---------------------------------------------------------------------------

T = 24 * 60 // 15        # 15-min intervals per day  →  96
K = 3                    # number of EV states: 0=off, 1=low, 2=high
STATE_NAMES = ["off", "low", "high"]


# ---------------------------------------------------------------------------
# Numeric / algorithmic constants
# ---------------------------------------------------------------------------

# HMM / EV-side
LAPLACE = 1e-3                 # smoothing for HMM transition counts
SIGMA_EV_OFF = 1e-3            # floor for off-state EV emission std
EM_TOL = 1e-6
EM_MAX_ITERS = 100
THETA_VAR_FLOOR = 1e-6

# Non-EV-side (PPCA + omega — current implementation, to be replaced in Stage 2)
PPCA_RANK_DEFAULT = 5          # rank r for Sigma_eta = W W^T + diag(psi)
PSI_FLOOR = 1e-6               # floor for per-t residual variance of eta prior
OMEGA2_FLOOR = 1e-6            # floor for per-home, per-t variance
SLICE_W = 1.5                  # slice sampler initial step (log-variance units)
SLICE_MAX_STEPS = 50           # safety cap on stepping-out iterations
SLICE_MAX_SHRINK = 50          # safety cap on shrinkage iterations
IG_MIN_SHAPE = 2.01            # a > 2 required for finite IG variance

# State-magnitude semantics for EV charging states (see specs/model.md §1.5).
# Encoded as truncation on the Theta^(n)_k prior. Off is pinned to 0; bound unused.
THETA_BOUNDS = [
    (0.0, 0.0),        # off    — Theta_off pinned to 0
    (0.1, 2.0),        # low    — 0.1 kW ≤ Theta_low ≤ 2 kW
    (2.0, np.inf),     # high   — Theta_high ≥ 2 kW
]


# ---------------------------------------------------------------------------
# ModelParams — all globally point-estimated parameters
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

    # EV state (§1)
    p_C: float                  # EV prevalence
    pi_z: np.ndarray            # (K,) initial EV-state probabilities at t=0
    P_z: np.ndarray             # (K, K) row-stochastic transition matrix

    # EV charging magnitudes (§1.5, §1.6)
    mu_theta:    np.ndarray     # (K,) per-state EV charging mean. mu_theta[0] = 0.
    sigma2_theta: np.ndarray    # (K,) per-state Theta prior variance. [0] = 0.
    sigma2_ev:   np.ndarray     # (K,) per-state EV emission variance. [0] = SIGMA_EV_OFF^2.

    # Non-EV: hierarchical prior on per-home mean profile eta^(n) ∈ R^T (§2.1–§2.2)
    eta_bar: np.ndarray         # (T,) global mean profile
    W_eta:   np.ndarray         # (T, r) PPCA loading matrix
    psi_eta: np.ndarray         # (T,) per-t residual variance

    # Non-EV variance: one of two parameterizations (§2.3–§2.4)
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
# HomeInference — per-home Gibbs output
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

    # Collapsed Gibbs only: marginal likelihood traces
    log_Z1_trace: np.ndarray | None = None       # (S_burn + S,) log p(x | C=1, α, Θ)
    log_Z0_trace: np.ndarray | None = None       # (S_burn + S,) log p(x | C=0, α)

    S_burn: int = 0                              # number of burn-in iterations
