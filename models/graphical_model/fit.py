"""Top-level fit() orchestrator.

Reads a fully-labeled training dataframe and runs the four fit steps in order,
returning a populated ModelParams. Each step delegates to a submodel module:

    Step 1: EV prevalence p_C            — closed form here
    Step 2: HMM parameters (pi_z, P_z)   — ev.fit_hmm
    Step 3: Non-EV LDS submodel          — non_ev_lds.fit_lds_em
    Step 4: Charging magnitudes          — ev.fit_charging_em

See specs/model.md §1–§2 for the math of each step.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import ev, non_ev_lds
from ._data import build_home_arrays
from .params import LDS_EM_MAX_ITERS, LDS_EM_TOL, T, ModelParams


def fit(
    train_df: pd.DataFrame,
    *,
    lds_init: non_ev_lds.LDSParams | None = None,
    lds_fit_A: bool = False,
    lds_fit_C: bool = False,
    lds_diagonal_Q: bool = True,
    lds_diagonal_R: bool = True,
    lds_diagonal_Sigma_0: bool = True,
    lds_max_iters: int = LDS_EM_MAX_ITERS,
    lds_tol: float = LDS_EM_TOL,
    verbose: bool = True,
) -> ModelParams:
    """Fit all global parameters from a fully-labeled training dataframe.

    Required columns: home_id, day, time_index, total_load, ev_load,
                      non_ev_load, charge_state, has_ev, city.

    LDS-related arguments (specs/model.md §2):
      lds_init             : optional warm-start LDSParams. Defaults to identity-diagonal.
      lds_fit_A, lds_fit_C : whether EM updates A, C (held at identity by default).
      lds_diagonal_*       : whether to constrain Q, R, Sigma_0 to be diagonal.
      lds_max_iters, lds_tol: EM stopping criteria.
    """
    if verbose:
        print("=" * 60)
        print("FIT: graphical model (LDS Non-EV submodel)")
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
    # Step 1 — EV prevalence p_C                                  (§1.2)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print("\n[Step 1] EV prevalence p_C")
    p_C = sorted_df.groupby("home_id")["has_ev"].first().mean()    # empirical mean — Bernoulli MLE
    if verbose:
        print(f"  p_C = {p_C:.4f} ({int(p_C * N)}/{N} homes have EV)")
        print(f"  Step 1 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Pre-shape per-home arrays for downstream steps
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Pre-shape] Building per-home (D, T) arrays")
    t0 = time.time()
    home_arrays = build_home_arrays(sorted_df, homes)              # df → {hid: {x, x_ev, x_nev, z, ...}}
    if verbose:
        ds = [a["D"] for a in home_arrays.values()]
        print(f"  D^(n) range: min={min(ds)}, median={int(np.median(ds))}, max={max(ds)}")
        print(f"  Pre-shape done in {time.time() - t0:.3f}s")

    ev_homes = [hid for hid in homes if home_arrays[hid]["has_ev"]]
    N_EV     = len(ev_homes)

    # ------------------------------------------------------------------
    # Step 2 — HMM parameters from EV homes only                  (§1.4)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 2] HMM parameters from {N_EV} EV homes")
    pi_z, P_z = ev.fit_hmm(home_arrays, ev_homes, verbose=verbose)  # Laplace-smoothed empirical counts
    if verbose:
        print(f"  Step 2 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 3 — Non-EV LDS submodel                                (§2)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 3] LDS Non-EV submodel from all {N} homes")
    nonev_obs_per_home = [home_arrays[hid]["x_nev"] for hid in homes]   # list of (D_n, T) arrays
    lds = non_ev_lds.fit_lds_em(
        nonev_obs_per_home,
        init=lds_init,
        latent_dim=T,
        fit_A=lds_fit_A, fit_C=lds_fit_C,
        diagonal_Q=lds_diagonal_Q,
        diagonal_R=lds_diagonal_R,
        diagonal_Sigma_0=lds_diagonal_Sigma_0,
        max_iters=lds_max_iters,
        tol=lds_tol,
        verbose=verbose,
    )
    if verbose:
        print(f"  Step 3 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 4 — EV charging magnitudes via EM                      (§1.6)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 4] EV charging magnitudes (EM) from {N_EV} EV homes")
    mu_theta, sigma2_theta, sigma2_ev = ev.fit_charging_em(
        home_arrays, ev_homes, verbose=verbose,
    )                                                              # one-way Gaussian RE model, EM to MLE
    if verbose:
        print(f"  Step 4 done in {time.time() - t0:.3f}s")

    params = ModelParams(
        p_C=float(p_C),
        pi_z=pi_z, P_z=P_z,
        mu_theta=mu_theta,
        sigma2_theta=sigma2_theta,
        sigma2_ev=sigma2_ev,
        lds=lds,
    )

    if verbose:
        print("\n" + params.summary())

    return params
