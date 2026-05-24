"""Per-home Gibbs samplers and dataset-level inference drivers.

Two per-home Gibbs samplers (specs/model.md §4):

    infer_home              — mixture-Gibbs on z, logistic heuristic for C  (legacy)
    infer_home_collapsed    — collapsed sampler: C drawn from exact marginal
                              posterior, then z|C. PREFERRED.

Both share the same conditional updates for the per-home latents (theta, eta,
omega^2); they only differ in how C is sampled. Block dispatch lives below;
each call site has a one-line inline comment explaining *how* that function
implements the stated update.

The latent decomposition x = x_EV + x_Non-EV stays marginalized throughout
(specs §2.5): we never sample x_EV, x_Non-EV separately.

Dataset-level drivers (`infer_all`, `infer_all_collapsed`) iterate over homes.
Convenience adapters for the heuristic-detector pipeline are at the bottom.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import ev, non_ev_ppca
from .params import HomeInference, K, ModelParams, STATE_NAMES, T


# ===========================================================================
# Likelihoods used by the C-step
# ===========================================================================

def compute_loglik(
    x: np.ndarray,         # (D, T)
    z: np.ndarray,         # (D, T)
    theta: np.ndarray,     # (K,)
    eta:   np.ndarray,     # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
) -> float:
    """Complete-data log-likelihood:

        Σ_{d,t} log N( x[d,t] ; theta[z[d,t]] + eta[t],
                                 sigma2_ev[z[d,t]] + omega2[t] )
    """
    var_dt  = params.sigma2_ev[z] + omega2[None, :]
    mean_dt = theta[z] + eta[None, :]
    ll = -0.5 * (np.log(2 * np.pi * var_dt) + (x - mean_dt) ** 2 / var_dt)
    return float(ll.sum())


def compute_loglik_c0(
    x: np.ndarray,         # (D, T)
    eta: np.ndarray,       # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
) -> float:
    """log p(x | C=0) = log p(x | z≡off):

        Σ_{d,t} log N( x[d,t] ; eta[t], sigma2_ev[off] + omega2[t] )
    """
    var_t    = params.sigma2_ev[0] + omega2
    residual = x - eta[None, :]
    ll = -0.5 * (np.log(2 * np.pi * var_t[None, :])
                 + residual ** 2 / var_t[None, :])
    return float(ll.sum())


# ===========================================================================
# Mixture-Gibbs sampler (legacy)
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
    """Mixture Gibbs for one home with hierarchical Non-EV submodel.

    home_x           : (D, T) total grid power — the only signal at test time.
    initial_c        : warm-start C value (0 or 1).
    initial_z        : (D, T) warm-start z; defaults to all-off if None.
    c_logistic_model : fitted sklearn LogisticRegression on transitions/day.
                       If None, falls back to a hard threshold at 1.0.

    Per-iteration block structure (specs/model.md §4.1):
      A.  Mixture-Gibbs z step (FFBS candidate vs all-off, weighted by p_C)
      B.  C | z via logistic-on-transitions heuristic
      2.  Theta_k for k ∈ {low, high}
      3.  eta            (T-dim conjugate Gaussian under PPCA prior)
      4.  omega^2        (slice sample per t — only in 'hierarchical' mode)
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
    omega2 = _initial_omega2(params)
    z      = initial_z.copy() if initial_z is not None else np.zeros((D, T), dtype=np.int64)
    c      = initial_c

    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    # Σ_η^{-1} is constant across the chain (depends only on W_eta, psi_eta)
    Sigma_eta_inv        = non_ev_ppca.compute_sigma_eta_inv(params.W_eta, params.psi_eta)
    Sigma_eta_inv_etabar = Sigma_eta_inv @ params.eta_bar

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
    n_slice_evals = 0

    for it in range(n_total):

        # Block A: mixture-Gibbs z step  — propose FFBS draw vs z≡off, pick by softmax
        z_candidate, log_Z1 = ev.ffbs(home_x, theta, eta, omega2, params,
                                       log_pi, log_P, rng)            # FFBS proposal under C=1
        log_Z0    = compute_loglik_c0(home_x, eta, omega2, params)    # marginal under z≡off
        log_w1    = np.log(params.p_C + 1e-300)     + log_Z1
        log_w0    = np.log(1 - params.p_C + 1e-300) + log_Z0
        p_c1_eff  = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))
        z = z_candidate if rng.random() < np.clip(p_c1_eff, 0.0, 1.0) \
            else np.zeros((D, T), dtype=np.int64)

        # Block B: C | z  — logistic on per-day transition rate (heuristic, see specs §1.1)
        transitions_per_day_now = float((np.diff(z, axis=1) != 0).sum() / D)
        if c_logistic_model is not None:
            p_c1 = float(c_logistic_model.predict_proba([[transitions_per_day_now]])[0, 1])
        else:
            p_c1 = float(transitions_per_day_now > 1.0)
        c = int(rng.random() < p_c1)

        # Block 2: Θ_k  — truncated-Normal × Gaussian conjugate, per state k (§1.5)
        for k in (1, 2):
            theta[k] = ev.sample_theta_k(home_x, z, eta, omega2, params, k, rng)

        # Block 3: η  — T-dim conjugate Gaussian under PPCA prior (§2.1); Cholesky of (Σ_η^{-1}+diag(λ_t))
        eta = non_ev_ppca.sample_eta(home_x, z, theta, omega2, params,
                                      Sigma_eta_inv, Sigma_eta_inv_etabar, rng)

        # Block 4: ω²  — slice sample per t in log-variance space (§2.3); skipped in 'global' mode
        if params.omega_mode == "hierarchical":
            omega2, evals_this_iter = non_ev_ppca.sample_omega(
                home_x, z, theta, eta, omega2, params, rng,
            )
            n_slice_evals += evals_this_iter

        # --- traces ---------------------------------------------------------
        if record_traces:
            eta_trace[it]       = eta
            omega2_trace[it]    = omega2
            theta_trace[it]     = theta
            state_occ_trace[it] = [(z == k).mean() for k in range(K)]
            loglik_trace[it]    = compute_loglik(home_x, z, theta, eta, omega2, params)

        # --- accumulate post-burn-in ----------------------------------------
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


# ===========================================================================
# Collapsed Gibbs sampler (preferred)
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
    """Collapsed Gibbs for one home — preferred default (specs/model.md §4.1).

    Each iteration:
      Block 1.  C | x, Θ, η, ω        — Bernoulli on softmax of {log p(x|C=0,…),
                                                                 log p(x|C=1,…)};
                                          the C=1 marginal comes from the HMM forward pass.
      Block 2.  z | C, x, Θ, η, ω     — backward sample if C=1, pin to off if C=0.
      Block 3.  Θ_k | z, η, ω, x      — truncated-Normal conjugate (identical to infer_home).
      Block 4.  η | z, Θ, ω, x        — T-dim conjugate Gaussian under PPCA prior.
      Block 5.  ω² | z, Θ, η, x       — slice sample per t (only in 'hierarchical' mode).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"

    if verbose:
        print(f"  [home {home_id}] D={D} → "
              f"collapsed Gibbs ({S_burn} burn-in + {S} retained)")

    # ── initialise ────────────────────────────────────────────────────────────
    theta  = params.mu_theta.copy()
    eta    = params.eta_bar.copy()
    omega2 = _initial_omega2(params)
    z      = np.zeros((D, T), dtype=np.int64)
    c      = 0

    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    Sigma_eta_inv        = non_ev_ppca.compute_sigma_eta_inv(params.W_eta, params.psi_eta)
    Sigma_eta_inv_etabar = Sigma_eta_inv @ params.eta_bar

    # ── storage ───────────────────────────────────────────────────────────────
    n_total                       = S_burn + S
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
        log_Z1_trace    = np.zeros(n_total,      dtype=np.float64)
        log_Z0_trace    = np.zeros(n_total,      dtype=np.float64)
    else:
        eta_trace = omega2_trace = theta_trace = state_occ_trace = loglik_trace = None
        log_Z1_trace = log_Z0_trace = None

    # ── main loop ─────────────────────────────────────────────────────────────
    t_start = time.time()
    s_idx   = -1

    for it in range(n_total):

        # Block 1: C  — collapsed posterior, marginalising z via the HMM forward pass
        log_f, log_Z1 = ev.hmm_forward(home_x, theta, eta, omega2, params,
                                        log_pi, log_P)              # log_Z1 = log p(x | C=1, …)
        log_Z0 = compute_loglik_c0(home_x, eta, omega2, params)     # log p(x | C=0, …) = z≡off
        log_w1 = np.log(params.p_C + 1e-300) + log_Z1
        log_w0 = np.log(1 - params.p_C + 1e-300) + log_Z0
        p_c1   = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))   # softmax in log-space
        c      = int(rng.random() < p_c1)

        # Block 2: z | C  — backward-sample under C=1, otherwise pin to off
        z = (ev.hmm_backward_sample(log_f, params, rng) if c == 1
             else np.zeros((D, T), dtype=np.int64))

        # Block 3: Θ_k  — truncated-Normal × Gaussian conjugate, per state k (§1.5)
        for k in (1, 2):
            theta[k] = ev.sample_theta_k(home_x, z, eta, omega2, params, k, rng)

        # Block 4: η  — T-dim conjugate Gaussian under PPCA prior (§2.1); Cholesky of (Σ_η^{-1}+diag(λ_t))
        eta = non_ev_ppca.sample_eta(home_x, z, theta, omega2, params,
                                      Sigma_eta_inv, Sigma_eta_inv_etabar, rng)

        # Block 5: ω²  — slice sample per t (§2.3); skipped in 'global' mode
        if params.omega_mode == "hierarchical":
            omega2, _ = non_ev_ppca.sample_omega(home_x, z, theta, eta, omega2, params, rng)

        # ── record ────────────────────────────────────────────────────────────
        if record_traces:
            eta_trace[it]       = eta
            omega2_trace[it]    = omega2
            theta_trace[it]     = theta
            state_occ_trace[it] = [(z == k).mean() for k in range(K)]
            loglik_trace[it]    = compute_loglik(home_x, z, theta, eta, omega2, params)
            log_Z1_trace[it]    = log_Z1
            log_Z0_trace[it]    = log_Z0

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

        if verbose and (it < 3 or it == S_burn or (it + 1) % 100 == 0):
            phase   = "burn-in" if it < S_burn else "keep  "
            elapsed = time.time() - t_start
            ll = loglik_trace[it] if record_traces else float("nan")
            print(f"    iter {it+1:4d}/{n_total} [{phase}]  "
                  f"C={c}  Θ_low={theta[1]:.3f}  Θ_high={theta[2]:.3f}  "
                  f"η∈[{eta.min():+.2f},{eta.max():+.2f}]  "
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
        log_Z1_trace                  = log_Z1_trace,
        log_Z0_trace                  = log_Z0_trace,
        S_burn                        = S_burn,
    )


# ===========================================================================
# Dataset-level drivers
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


def infer_all_collapsed(
    df: pd.DataFrame,
    params: ModelParams,
    *,
    S_burn: int = 200,
    S: int = 500,
    seed: int = 0,
    verbose: bool = True,
) -> dict[int, HomeInference]:
    """Run infer_home_collapsed on every home in df."""
    if verbose:
        print("=" * 60)
        print("INFERENCE: collapsed Gibbs over all homes")
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
# Heuristic adapters (used to bridge the EV-detection heuristic and the Gibbs sampler)
# ===========================================================================

def c_prob_from_z_via_heuristic(
    inference: HomeInference,
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
# Small helpers
# ===========================================================================

def _initial_omega2(params: ModelParams) -> np.ndarray:
    """Initial ω² for the Gibbs chain. DISPATCH on omega_mode (specs §2.3)."""
    if params.omega_mode == "global":
        # Fixed across all iterations; no Gibbs block updates it
        return params.sigma2_nev_global.copy()
    if params.omega_mode == "hierarchical":
        # IG mode b/(a+1) is slightly more conservative than the mean b/(a-1)
        return (params.b_omega / (params.a_omega + 1.0)).copy()
    raise ValueError(f"unknown omega_mode={params.omega_mode!r}")
