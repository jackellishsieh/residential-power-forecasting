"""Per-home collapsed Gibbs sampler + dataset-level driver.

The Non-EV submodel is the per-home daily LDS (`non_ev_lds.py`,
specs/model.md §2). The EV side (z^EV, Theta, HMM) is unchanged from §1.
Three-block collapsed Gibbs (specs §4):

    Block 1.  (C, z^EV) | z^LDS, Theta            — collapsed C, then z^EV | C
              Conditional on z^LDS the Non-EV offset (C z^LDS)[d, t] is known,
              so the HMM forward pass over z^EV computes
                  log p(x | C=1, z^LDS, Theta).
              C is drawn from the exact Bernoulli posterior using that and the
              z≡off marginal log p(x | C=0, z^LDS).  z^EV is backward-sampled
              if C=1 and pinned to off if C=0.
    Block 2.  Theta_k | z^EV, z^LDS, x            — truncated-Normal conjugate
              per state k ∈ {low, high}, treating (C z^LDS)[d, t] as the
              Non-EV offset and diag(R)[t] as the per-t Non-EV variance.
    Block 3.  z^LDS | z^EV, Theta, x              — Kalman FFBS on residuals
              x - theta[z^EV] with per-cell extra obs noise sigma2_ev[z^EV].

The legacy mixture-Gibbs sampler that lived here (with the logistic-on-
transitions C-step) has been removed: it was tied to the deprecated PPCA
Non-EV submodel and never used by the LDS pipeline.  The collapsed sampler
mixes well on its own (specs §4.1).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import ev, non_ev_lds
from .params import HomeInference, K, ModelParams, STATE_NAMES, T


# ===========================================================================
# Likelihoods used by the C-step
# ===========================================================================

def compute_loglik(
    x:          np.ndarray,     # (D, T)
    z:          np.ndarray,     # (D, T)
    theta:      np.ndarray,     # (K,)
    nonev_mean: np.ndarray,     # (D, T)   per-cell Non-EV offset, e.g. (C z^LDS)[d, t]
    nonev_var:  np.ndarray,     # (T,)     per-t Non-EV emission variance, e.g. diag(R)[t]
    params: ModelParams,
) -> float:
    """Complete-data log-likelihood given z^EV and the Non-EV offset:

        Σ_{d,t} log N( x[d,t] ; theta[z[d,t]] + nonev_mean[d,t],
                                sigma2_ev[z[d,t]] + nonev_var[t] )
    """
    var_dt  = params.sigma2_ev[z] + nonev_var[None, :]
    mean_dt = theta[z] + nonev_mean
    ll = -0.5 * (np.log(2 * np.pi * var_dt) + (x - mean_dt) ** 2 / var_dt)
    return float(ll.sum())


def compute_loglik_c0(
    x:          np.ndarray,     # (D, T)
    nonev_mean: np.ndarray,     # (D, T)
    nonev_var:  np.ndarray,     # (T,)
    params: ModelParams,
) -> float:
    """log p(x | C=0, z^LDS, Theta) = log p(x | z≡off, z^LDS):

        Σ_{d,t} log N( x[d,t] ; nonev_mean[d,t],  sigma2_ev[off] + nonev_var[t] )
    """
    var_t    = params.sigma2_ev[0] + nonev_var          # (T,)
    residual = x - nonev_mean                           # (D, T)
    ll = -0.5 * (np.log(2 * np.pi * var_t[None, :])
                 + residual ** 2 / var_t[None, :])
    return float(ll.sum())


# ===========================================================================
# Per-home collapsed Gibbs sampler
# ===========================================================================

def infer_home_collapsed(
    home_x: np.ndarray,
    params: ModelParams,
    *,
    S_burn: int = 200,
    S: int = 500,
    rng: np.random.Generator | None = None,
    home_id: int = -1,
    verbose: bool = True,
    record_traces: bool = True,
) -> HomeInference:
    """Collapsed Gibbs for one home — specs/model.md §4.1.

    home_x : (D, T) total grid power — the only signal at test time.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"

    if verbose:
        print(f"  [home {home_id}] D={D} → "
              f"collapsed Gibbs ({S_burn} burn-in + {S} retained)")

    lds       = params.lds
    C_lds     = lds.C                                   # (T, L)  emission
    nonev_var = np.diag(lds.R).copy()                   # (T,)    per-t Non-EV variance

    # ── initialise ────────────────────────────────────────────────────────────
    theta = params.mu_theta.copy()
    # Warm-start z^LDS from the Kalman smoother on home_x: treats x as if it were
    # all Non-EV (i.e. the C=0 hypothesis). This costs one smoother call up front;
    # subsequent iterations refine it conditional on z^EV.
    z_lds = lds.smooth(home_x).z_smooth                 # (D, L)
    z     = np.zeros((D, T), dtype=np.int64)
    c     = 0

    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    # ── storage ───────────────────────────────────────────────────────────────
    n_total                       = S_burn + S
    z_counts                      = np.zeros((D, T, K), dtype=np.float64)
    theta_samples                 = np.zeros((S, K),    dtype=np.float64)
    c_samples                     = np.zeros(S,         dtype=np.int8)
    c_from_z_samples              = np.zeros(S,         dtype=np.int8)
    z_transitions_per_day_samples = np.zeros(S,         dtype=np.float64)
    z_lds_mean                    = np.zeros((D, T),    dtype=np.float64)   # accumulated

    if record_traces:
        theta_trace     = np.zeros((n_total, K), dtype=np.float64)
        state_occ_trace = np.zeros((n_total, K), dtype=np.float64)
        loglik_trace    = np.zeros(n_total,      dtype=np.float64)
        log_Z1_trace    = np.zeros(n_total,      dtype=np.float64)
        log_Z0_trace    = np.zeros(n_total,      dtype=np.float64)
    else:
        theta_trace = state_occ_trace = loglik_trace = None
        log_Z1_trace = log_Z0_trace = None

    # ── main loop ─────────────────────────────────────────────────────────────
    t_start  = time.time()
    z_lds_last: np.ndarray | None = None

    for it in range(n_total):

        # Per-cell Non-EV offset under current z^LDS sample.
        nonev_mean = z_lds @ C_lds.T                                # (D, T) — equals z_lds when C=I

        # Block 1: C  — collapsed posterior, marginalising z^EV via HMM forward pass.
        log_f, log_Z1 = ev.hmm_forward(home_x, theta, nonev_mean, nonev_var,
                                        params, log_pi, log_P)      # log_Z1 = log p(x | C=1, z^LDS, Θ)
        log_Z0 = compute_loglik_c0(home_x, nonev_mean, nonev_var,    # log p(x | C=0, z^LDS)
                                    params)
        log_w1 = np.log(params.p_C       + 1e-300) + log_Z1
        log_w0 = np.log(1 - params.p_C   + 1e-300) + log_Z0
        p_c1   = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))
        c      = int(rng.random() < p_c1)

        # Block 1b: z^EV | C   — backward-sample if C=1, pin to off if C=0.
        z = (ev.hmm_backward_sample(log_f, params, rng) if c == 1
             else np.zeros((D, T), dtype=np.int64))

        # Block 2: Θ_k  — truncated-Normal × Gaussian conjugate, per state k.
        for k in (1, 2):
            theta[k] = ev.sample_theta_k(home_x, z, nonev_mean, nonev_var,
                                          params, k, rng)

        # Block 3: z^LDS  — Kalman FFBS on residuals x - Θ_z, heteroscedastic per cell.
        z_lds = non_ev_lds.sample_z_lds(home_x, z, theta, params.sigma2_ev,
                                         lds, rng)                  # (D, L)

        # ── record ────────────────────────────────────────────────────────────
        if record_traces:
            theta_trace[it]     = theta
            state_occ_trace[it] = [(z == k).mean() for k in range(K)]
            loglik_trace[it]    = compute_loglik(home_x, z, theta, nonev_mean,
                                                  nonev_var, params)
            log_Z1_trace[it]    = log_Z1
            log_Z0_trace[it]    = log_Z0

        if it >= S_burn:
            s_idx                 = it - S_burn
            theta_samples[s_idx]  = theta
            c_samples[s_idx]      = c
            for k in range(K):
                z_counts[:, :, k] += (z == k)
            c_from_z_samples[s_idx]              = int(np.any(z != 0))
            z_transitions_per_day_samples[s_idx] = float(
                (np.diff(z, axis=1) != 0).sum() / D
            )
            # Running mean of z^LDS as (C z^LDS)[d, t] in observation space.
            z_lds_mean += (z_lds @ C_lds.T - z_lds_mean) / (s_idx + 1)
            z_lds_last  = (z_lds @ C_lds.T).copy()

        if verbose and (it < 3 or it == S_burn or (it + 1) % 100 == 0):
            phase   = "burn-in" if it < S_burn else "keep  "
            elapsed = time.time() - t_start
            ll = loglik_trace[it] if record_traces else float("nan")
            print(f"    iter {it+1:4d}/{n_total} [{phase}]  "
                  f"C={c}  Θ_low={theta[1]:.3f}  Θ_high={theta[2]:.3f}  "
                  f"logL={ll:.1f}  ({elapsed:.1f}s)")

    # ── summaries ─────────────────────────────────────────────────────────────
    z_marginals = z_counts / S
    z_hat       = np.argmax(z_marginals, axis=2)
    c_hat_prob  = float(c_samples.mean())

    if verbose:
        elapsed = time.time() - t_start
        frac = z_marginals.mean(axis=(0, 1))
        print(f"\n  [home {home_id}] done in {elapsed:.1f}s")
        print(f"    P̂(C=1) from chain : {c_hat_prob:.4f}  (hard={int(c_hat_prob >= 0.5)})")
        print(f"    z freq : off={frac[0]:.3f}  low={frac[1]:.3f}  high={frac[2]:.3f}")
        for k in (1, 2):
            print(f"    Θ[{STATE_NAMES[k]:>4}] : "
                  f"mean={theta_samples[:,k].mean():.3f}  "
                  f"std={theta_samples[:,k].std():.4f}")

    return HomeInference(
        home_id                       = home_id,
        C_hat                         = int(c_hat_prob >= 0.5),
        z_hat                         = z_hat,
        z_marginals                   = z_marginals,
        theta_samples                 = theta_samples,
        z_lds_mean                    = z_lds_mean,
        z_lds_last                    = z_lds_last,
        c_samples                     = c_samples,
        c_from_z_samples              = c_from_z_samples,
        z_transitions_per_day_samples = z_transitions_per_day_samples,
        theta_trace                   = theta_trace,
        state_occ_trace               = state_occ_trace,
        loglik_trace                  = loglik_trace,
        log_Z1_trace                  = log_Z1_trace,
        log_Z0_trace                  = log_Z0_trace,
        S_burn                        = S_burn,
    )


# ===========================================================================
# Dataset-level driver
# ===========================================================================

def infer_all_collapsed(
    df: pd.DataFrame,
    params: ModelParams,
    *,
    S_burn: int = 200,
    S: int = 500,
    seed: int = 0,
    verbose: bool = True,
) -> dict[int, HomeInference]:
    """Run `infer_home_collapsed` on every home in df."""
    if verbose:
        print("=" * 60)
        print("INFERENCE: collapsed Gibbs over all homes (LDS Non-EV submodel)")
        print("=" * 60)

    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    homes     = list(sorted_df["home_id"].unique())
    rng       = np.random.default_rng(seed)

    results: dict[int, HomeInference] = {}
    t0 = time.time()
    for i, hid in enumerate(homes):
        g = sorted_df[sorted_df["home_id"] == hid]
        D = len(g) // T
        x = g["total_load"].to_numpy().reshape(D, T).astype(np.float64)

        if verbose:
            true_c = int(g["has_ev"].iloc[0]) if "has_ev" in g.columns else "?"
            print(f"\n[{i+1}/{len(homes)}] home {hid}  D={D}  true_c={true_c}")

        results[int(hid)] = infer_home_collapsed(
            x, params,
            S_burn=S_burn, S=S,
            rng=rng, home_id=int(hid), verbose=verbose,
        )

    if verbose:
        print(f"\nAll homes done in {time.time() - t0:.1f}s")

    return results


# ===========================================================================
# Heuristic adapter
# ===========================================================================

def build_heuristic_homes(df: pd.DataFrame) -> dict:
    """Reconstruct {dataid: (has_car, city, df_with_load_col)} from flat train_df.

    Format expected by `notebooks/utils/first_diff_logistic` — the heuristic
    C-detector used as a comparison baseline in the inference notebook.
    """
    out = {}
    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    for hid, g in sorted_df.groupby("home_id", sort=False):
        out[int(hid)] = (
            bool(g["has_ev"].iloc[0]),
            g["city"].iloc[0],
            g[["total_load"]].rename(columns={"total_load": "load"}).reset_index(drop=True),
        )
    return out
