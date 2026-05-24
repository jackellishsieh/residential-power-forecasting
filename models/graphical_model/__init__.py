"""Generative graphical model for residential power: fit + Gibbs inference.

Notation and derivations follow specs/model.md. Briefly:

  Per-home/day/timestep emission:

      x^(n)_{d,t} | z^(n)_{d,t}=k  ~  N( Theta^(n)_k + eta^(n)_t,
                                          (sigma^EV_k)^2 + omega^2_t(n,...) )

  with a hierarchical prior across homes on the per-home Non-EV mean
  profile eta^(n) ∈ R^T:

      eta^(n) ~ N( eta_bar, Sigma_eta = W W^T + diag(psi) )      (PPCA / FA)

  and one of two parameterizations for the Non-EV variance profile,
  controlled by `omega_mode` in ModelParams.

Package layout — each module mirrors a section of specs/model.md:

    params.py           constants, ModelParams, HomeInference     (§0)
    _data.py            df → per-home (D, T) arrays
    ev.py               EV submodel: HMM, charging magnitudes     (§1)
    non_ev_ppca.py      Non-EV submodel: PPCA eta + omega         (§2)
    fit.py              top-level fit() orchestrator
    inference.py        Gibbs samplers + dataset-level drivers    (§4)
    evaluation.py       confusion matrices, printing              (§5)

Re-exports below give a stable public API; existing notebooks importing
`from models import graphical_model as gm` continue to work.
"""

from .evaluation import evaluate, format_confusion, print_evaluation
from .fit import fit
from .inference import (
    build_heuristic_homes,
    c_prob_from_z_via_heuristic,
    compute_loglik,
    compute_loglik_c0,
    infer_all,
    infer_all_collapsed,
    infer_home,
    infer_home_collapsed,
)
from .params import (
    IG_MIN_SHAPE,
    K,
    LAPLACE,
    OMEGA2_FLOOR,
    PPCA_RANK_DEFAULT,
    PSI_FLOOR,
    SIGMA_EV_OFF,
    SLICE_MAX_SHRINK,
    SLICE_MAX_STEPS,
    SLICE_W,
    STATE_NAMES,
    T,
    THETA_BOUNDS,
    THETA_VAR_FLOOR,
    HomeInference,
    ModelParams,
)

__all__ = [
    # Public dataclasses / constants
    "ModelParams", "HomeInference", "STATE_NAMES", "T", "K",
    "PPCA_RANK_DEFAULT", "THETA_BOUNDS",
    # Fit
    "fit",
    # Inference
    "infer_home", "infer_home_collapsed",
    "infer_all",  "infer_all_collapsed",
    "compute_loglik", "compute_loglik_c0",
    "c_prob_from_z_via_heuristic", "build_heuristic_homes",
    # Evaluation
    "evaluate", "print_evaluation", "format_confusion",
]
