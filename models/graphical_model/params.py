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
# Per-home inference outputs — one container per EV hypothesis, plus the
# combined comparison result.  See models/graphical_model/inference.py.
# ---------------------------------------------------------------------------

@dataclass
class HomeInferenceC0:
    """Exact C=0 (no-EV) inference output.

    Under C=0 the only latent is z^LDS, whose posterior is Gaussian and exact
    (one Kalman smoother call). `log_lik` is the Kalman marginal log p(x | C=0)
    with z^LDS integrated out; `log_evidence` adds the log(1 - p_C) prior.
    """
    home_id: int
    z_lds_mean:     np.ndarray                   # (D, T) E[C z^LDS | x, C=0]  (obs space)
    z_lds_cov_diag: np.ndarray                   # (D, T) Var diag of the Non-EV mean per cell
    log_lik:        float                        # log p(x | C=0)
    log_evidence:   float                        # log p(x, C=0) = log(1-p_C) + log_lik


@dataclass
class HomeInferenceC1:
    """C=1 (EV) inference output — three-block Gibbs (Theta, z^EV, z^LDS).

    Memory note: the per-iter z^LDS is (D, T) ≈ 35 000 floats; we accumulate the
    posterior mean (`z_lds_mean`) incrementally and keep only the final retained
    sample (`z_lds_last`). z^EV samples are retained only if explicitly requested
    (`z_ev_samples`, needed for the Chib estimator B).
    """
    home_id: int
    z_hat:        np.ndarray                       # (D, T) MAP charging states (EV)
    z_marginals:  np.ndarray                       # (D, T, K) p(z_ev | x, C=1)
    theta_samples: np.ndarray                      # (S, K)
    theta_mean:   np.ndarray                       # (K,)
    z_lds_mean:   np.ndarray                        # (D, T) posterior mean of C z^LDS (accumulated)
    z_lds_last:   np.ndarray | None = None          # (D, T) last retained C z^LDS sample
    z_ev_last:    np.ndarray | None = None          # (D, T) last retained z^EV sample
    z_lds_star:   np.ndarray | None = None          # (D, T) conditional smoother mean at (z_hat, theta_mean)

    # ---- Model-comparison terms (specs §5) -----------------------------
    log_joint_plugin:    float = 0.0               # log p(x, z^EV*, z^LDS* | C=1, Θ*)  (A, no C prior)
    log_evidence_plugin: float = 0.0               # log p(x, C=1) via plug-in joint    (A)
    log_evidence_rb:     float = 0.0               # log p(x, C=1) with z^LDS integrated (A′; bridge to B)
    log_evidence_chib:   float | None = None       # log p(x, C=1) exact via Chib        (B; if computed)
    chib:                "ChibResult | None" = None  # full Chib breakdown (if computed)

    # ---- Full iteration traces (burn-in + retained) --------------------
    theta_trace:     np.ndarray | None = None      # (S_burn+S, K)
    state_occ_trace: np.ndarray | None = None      # (S_burn+S, K)
    loglik_trace:    np.ndarray | None = None       # (S_burn+S,)

    # ---- Retained z^EV samples (only if retain_z_ev=True) --------------
    z_ev_samples: np.ndarray | None = None         # (S, D, T) int8 — for Chib (B)

    S_burn: int = 0                                # number of burn-in iterations


@dataclass
class ChibResult:
    """Breakdown of the Chib (1995) estimate of log p(x | C=1) (specs §5, B).

    Identity at the high-density point ψ* = (z^EV*, Θ*, z^LDS*), all | C=1:

        log p(x|C=1) = log p(x|ψ*) + log p(ψ*) − log p(ψ*|x)

    with the posterior ordinate split by Gibbs block order (z^EV, Θ, z^LDS):

        log p(ψ*|x) = log p(z^EV*|x) + log p(Θ*|x,z^EV*) + log p(z^LDS*|x,z^EV*,Θ*)
                      └ ord_zev (main run) ┘ └ ord_theta (reduced) ┘ └ ord_zlds (closed form)
    """
    log_lik_star:    float    # log p(x | z^EV*, z^LDS*, Θ*)  (complete-data emission)
    log_prior_star:  float    # log p(z^EV*) + log p(Θ*) + log p(z^LDS*)
    ord_zev:         float    # log p(z^EV* | x)               — main-run average
    ord_theta:       float    # log p(Θ* | x, z^EV*)           — reduced-run average
    ord_zlds:        float    # log p(z^LDS* | x, z^EV*, Θ*)   — closed-form FFBS density
    log_lik_c1:      float    # log p(x | C=1) = log_lik_star + log_prior_star − (ords)
    log_evidence:    float    # log p(x, C=1) = log p_C + log_lik_c1
    # Cross-check: the Gaussian sub-identity should reproduce lds_loglik_c1_given_zev.
    lds_marg_direct: float    # lds_loglik_c1_given_zev(x, z^EV*, Θ*)
    lds_marg_via_chib: float  # log_lik_star + log p(z^LDS*) − ord_zlds  (should ≈ direct)


@dataclass
class HomeResult:
    """Combined per-home result: both tracks + the C=0 vs C=1 comparison.

    Exposes `.z_hat` / `.z_marginals` (from the C=1 track, for z-state evaluation)
    and `.c_prob` (soft P(C=1 | x) from the two evidences) so the evaluation code
    can treat it like the old single-track output.
    """
    home_id: int
    c0: HomeInferenceC0
    c1: HomeInferenceC1
    C_hat: int                                     # argmax of the two evidences
    log_evidence_c0: float
    log_evidence_c1: float                         # the estimate used for the decision
    decision: str = "rb"                           # which C=1 estimate drove the decision

    @property
    def z_hat(self) -> np.ndarray:
        """MAP charging states from the C=1 track (charging-state recovery)."""
        return self.c1.z_hat

    @property
    def z_marginals(self) -> np.ndarray:
        return self.c1.z_marginals

    @property
    def c_prob(self) -> float:
        """Soft P(C=1 | x) = softmax over the two log-evidences."""
        m = max(self.log_evidence_c0, self.log_evidence_c1)
        w0 = np.exp(self.log_evidence_c0 - m)
        w1 = np.exp(self.log_evidence_c1 - m)
        return float(w1 / (w0 + w1))
