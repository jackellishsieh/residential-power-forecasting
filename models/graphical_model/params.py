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

from .non_ev_lds import LDSParams


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

# Non-EV-side: LDS EM defaults
LDS_EM_TOL = 1e-4
LDS_EM_MAX_ITERS = 50

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

      The Non-EV submodel is the per-home daily LDS — its parameters live in
      `lds` (an `LDSParams` object). See models/graphical_model/non_ev_lds.py.
    """

    # EV state (§1)
    p_C: float                  # EV prevalence
    pi_z: np.ndarray            # (K,) initial EV-state probabilities at t=0
    P_z: np.ndarray             # (K, K) row-stochastic transition matrix

    # EV charging magnitudes (§1.5, §1.6)
    mu_theta:    np.ndarray     # (K,) per-state EV charging mean. mu_theta[0] = 0.
    sigma2_theta: np.ndarray    # (K,) per-state Theta prior variance. [0] = 0.
    sigma2_ev:   np.ndarray     # (K,) per-state EV emission variance. [0] = SIGMA_EV_OFF^2.

    # Non-EV submodel — per-home daily LDS (§2 of revised specs/model.md)
    lds: LDSParams

    K: int = K
    T: int = T

    @property
    def latent_dim(self) -> int:
        return self.lds.latent_dim

    def summary(self) -> str:
        lines = [
            "ModelParams summary",
            "=" * 40,
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

        lines.append("\nNon-EV — per-home daily LDS")
        lines.append(self.lds.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HomeInference — per-home Gibbs output
# ---------------------------------------------------------------------------

@dataclass
class HomeInference:
    """Per-home output of the Gibbs sampler.

    Memory note: the per-iter LDS latent z^LDS is (D, T) ≈ 35 000 floats.
    Storing all S samples would balloon memory, so we instead accumulate the
    posterior mean (`z_lds_mean`) incrementally and keep only the final-iter
    sample (`z_lds_last`) for visualization. Same for the predictive Non-EV
    mean E[C z^LDS_d] — accumulated, not stored per-iter.
    """
    home_id: int
    C_hat: int
    z_hat: np.ndarray                            # (D, T) MAP charging states (EV)

    # ---- Per-sample posterior summaries --------------------------------
    z_marginals:   np.ndarray | None = None      # (D, T, K) p(z_ev | x)
    theta_samples: np.ndarray | None = None      # (S, K)
    z_lds_mean:    np.ndarray | None = None      # (D, T) posterior mean of z^LDS (accumulated)
    z_lds_last:    np.ndarray | None = None      # (D, T) last retained sample of z^LDS

    # ---- Per-sample C draws & helpers ----------------------------------
    c_samples:                     np.ndarray | None = None    # (S,)  int {0,1}
    c_from_z_samples:              np.ndarray | None = None    # (S,)  any-nonoff indicator
    z_transitions_per_day_samples: np.ndarray | None = None    # (S,)  float

    # ---- Full iteration traces (burn-in + retained) --------------------
    theta_trace:    np.ndarray | None = None     # (S_burn+S, K)
    state_occ_trace:np.ndarray | None = None     # (S_burn+S, K)
    loglik_trace:   np.ndarray | None = None     # (S_burn+S,)
    log_Z1_trace:   np.ndarray | None = None     # (S_burn+S,) — collapsed sampler only
    log_Z0_trace:   np.ndarray | None = None     # (S_burn+S,) — collapsed sampler only

    S_burn: int = 0                              # number of burn-in iterations
