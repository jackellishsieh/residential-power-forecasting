"""Top-level fit() orchestrator.

Reads a fully-labeled training dataframe and runs the four fit steps in order,
returning a populated ModelParams. Each step delegates to a submodel module:

    Step 1: EV prevalence p_C            — closed form here
    Step 2: HMM parameters (pi_z, P_z)   — ev.fit_hmm
    Step 3: Non-EV submodel              — non_ev_ppca.fit_background
    Step 4: Charging magnitudes          — ev.fit_charging_em

See specs/model.md §1–§2 for the math of each step.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import ev, non_ev_ppca
from ._data import build_home_arrays
from .params import PPCA_RANK_DEFAULT, T, ModelParams


def fit(
    train_df: pd.DataFrame,
    *,
    ppca_rank: int = PPCA_RANK_DEFAULT,
    omega_mode: str = "global",
    verbose: bool = True,
) -> ModelParams:
    """Fit all global parameters from a fully-labeled training dataframe.

    Required columns: home_id, day, time_index, total_load, ev_load,
                      non_ev_load, charge_state, has_ev, city.

    ppca_rank  : rank r for the PPCA prior covariance of eta^(n).
                 r=0 corresponds to a plain diagonal prior diag(psi).
    omega_mode : Non-EV variance parameterization.
                 "global"       — fit a single T-vector sigma2_nev_global across
                                  homes; FIXED at inference. (DEFAULT.)
                 "hierarchical" — per-home (omega^(n)_t)^2 with IG prior;
                                  sampled at inference.
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
    # Step 3 — Hierarchical Non-EV submodel                     (§2.1–§2.4)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 3] Non-EV submodel from all {N} homes  "
              f"(PPCA rank r={ppca_rank}, omega_mode={omega_mode!r})")
    eta_bar, W_eta, psi_eta, sigma2_nev_global, a_omega, b_omega = non_ev_ppca.fit_background(
        home_arrays, list(homes),
        ppca_rank=ppca_rank, omega_mode=omega_mode, verbose=verbose,
    )                                                              # eta_bar+W+psi via truncated-eigen FA;
                                                                   # omega via global mean OR IG MoM
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
        eta_bar=eta_bar, W_eta=W_eta, psi_eta=psi_eta,
        omega_mode=omega_mode,
        sigma2_nev_global=sigma2_nev_global,
        a_omega=a_omega, b_omega=b_omega,
        mu_theta=mu_theta,
        sigma2_theta=sigma2_theta,
        sigma2_ev=sigma2_ev,
    )

    if verbose:
        print("\n" + params.summary())

    return params
