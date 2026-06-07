"""Generative graphical model for residential power: fit + Gibbs inference.

Notation and derivations follow specs/model.md. Briefly:

  Per-home/day/timestep emission:

      x^(n)_{d,t} | z^EV_{d,t}=k, z^LDS_d  ~
          N( Theta^(n)_k + (C z^LDS_d)[t],  (sigma^EV_k)^2 + R[t,t] )

  The Non-EV component evolves as a per-home daily linear dynamical system
  (specs §2, this branch):

      z^LDS_1      ~  N(mu_0, Sigma_0)
      z^LDS_d      ~  N(A z^LDS_{d-1}, Q)         d ≥ 2
      x^Non-EV_d   ~  N(C z^LDS_d, R)

  The LDS parameters (A, C, Q, R, mu_0, Sigma_0) are global; only the latent
  sequence z^LDS_{1:D} is per-home.

Package layout — each module mirrors a section of specs/model.md:

    params.py           constants, ModelParams, HomeInference     (§0)
    _data.py            df → per-home (D, T) arrays
    ev.py               EV submodel: HMM, charging magnitudes     (§1)
    non_ev_lds.py       Non-EV submodel: per-home daily LDS        (§2)
    fit.py              top-level fit() orchestrator
    inference.py        Gibbs samplers + dataset-level drivers    (§4)
    evaluation.py       confusion matrices, printing              (§5)

Re-exports below give a stable public API; existing notebooks importing
`from models import graphical_model as gm` continue to work for unchanged
public names (fit, infer_all_collapsed, ModelParams, etc.).
"""

from .evaluation import evaluate, format_confusion, print_evaluation
from .fit import fit
from .inference import (
    build_heuristic_homes,
    chib_marginal_loglik_c1,
    compute_loglik,
    compute_loglik_c0,
    infer_all,
    infer_home,
    infer_home_c0,
    infer_home_c1,
    lds_loglik_c0,
    lds_loglik_c1_given_zev,
    log_hmm_prior,
    log_joint_c1_plugin,
    log_lds_prior,
    log_theta_prior,
)
from .non_ev_lds import (
    LDSParams,
    fit_lds_em,
    kalman_filter,
    kalman_logpdf,
    kalman_sample,
    rts_smooth,
    sample_z_lds,
)
from .params import (
    K,
    LAPLACE,
    LDS_EM_MAX_ITERS,
    LDS_EM_TOL,
    SIGMA_EV_OFF,
    STATE_NAMES,
    T,
    THETA_BOUNDS,
    THETA_VAR_FLOOR,
    ChibResult,
    HomeInferenceC0,
    HomeInferenceC1,
    HomeResult,
    ModelParams,
)

__all__ = [
    # Public dataclasses / constants
    "ModelParams", "HomeInferenceC0", "HomeInferenceC1", "HomeResult", "ChibResult",
    "LDSParams", "STATE_NAMES", "T", "K", "THETA_BOUNDS",
    # Fit
    "fit", "fit_lds_em",
    # Inference — two-track (C=0 exact, C=1 Gibbs) + comparison
    "infer_home", "infer_all", "infer_home_c0", "infer_home_c1",
    "compute_loglik", "compute_loglik_c0",
    "lds_loglik_c0", "lds_loglik_c1_given_zev",
    "log_hmm_prior", "log_lds_prior", "log_joint_c1_plugin", "log_theta_prior",
    "chib_marginal_loglik_c1",
    "build_heuristic_homes",
    # LDS internals (useful in validation notebooks)
    "kalman_filter", "rts_smooth", "kalman_sample", "kalman_logpdf", "sample_z_lds",
    # Evaluation
    "evaluate", "print_evaluation", "format_confusion",
]
