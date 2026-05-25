"""Per-home Gibbs samplers and dataset-level inference drivers.

Two per-home Gibbs samplers (specs/model.md §4):

    infer_home              — legacy mixture-Gibbs on z_ev, logistic heuristic for C
    infer_home_collapsed    — collapsed sampler: C drawn from exact marginal
                              posterior, then z_ev | C. PREFERRED default.

Both share the same conditional updates for the per-home latents (theta, z_lds);
they only differ in how C and z_ev are sampled. Each Gibbs block is its own
small function: read the main loop body to see the block sequence, then drill
into the block function for the math.

Module structure:

    Section 1.  GibbsState            — per-home latent state container
    Section 2.  Likelihoods           — for the C-step and trace
    Section 3.  Gibbs blocks          — one function per block
                  gibbs_block_c_collapsed
                  gibbs_block_c_and_z_logistic
                  gibbs_block_theta
                  gibbs_block_z_lds
    Section 4.  Storage helpers       — buffer allocation, recording, accumulation
    Section 5.  Per-home samplers     — infer_home, infer_home_collapsed
    Section 6.  Dataset-level drivers — infer_all, infer_all_collapsed
    Section 7.  Heuristic adapters    — bridges to the C-detection heuristic
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import ev, non_ev_lds
from .params import HomeInference, K, ModelParams, STATE_NAMES, T


# ===========================================================================
# Section 1 — GibbsState
# ===========================================================================

@dataclass
class GibbsState:
    """All per-home latents being sampled, plus cached forward-pass outputs.

    Convention: every Gibbs block reads this state, samples one field
    conditional on the rest, and writes the new value back into the same
    object (in-place mutation for simplicity).
    """
    # Latents
    c:     int                          # EV ownership indicator
    z_ev:  np.ndarray                   # (D, T)  charging states {off, low, high}
    z_lds: np.ndarray                   # (D, T)  Non-EV latent (LDS state per day)
    theta: np.ndarray                   # (K,)    EV charging magnitudes

    # Cached forward-pass output from the most recent C-step (collapsed sampler)
    log_f:  np.ndarray | None = None    # (D, T, K) HMM filter messages (for backward sample)
    log_Z1: float = 0.0                 # log p(x | C=1, …) most-recent value
    log_Z0: float = 0.0                 # log p(x | C=0, …) most-recent value


def _initial_state(D: int, params: ModelParams) -> GibbsState:
    """Cold-start state: theta at hyperparameter mean, z_ev all-off,
    z_lds broadcast to mu_0 across all days."""
    z_lds = np.broadcast_to(params.lds.mu_0, (D, T)).copy()
    return GibbsState(
        c     = 0,
        z_ev  = np.zeros((D, T), dtype=np.int64),
        z_lds = z_lds,
        theta = params.mu_theta.copy(),
    )


# ===========================================================================
# Section 2 — Likelihoods (used by the C-step and the loglik trace)
# ===========================================================================

def compute_loglik(state: GibbsState, x: np.ndarray, params: ModelParams) -> float:
    """Complete-data log-likelihood of x given the current state:

        Σ_{d,t} log N( x[d,t] ; theta[z_ev[d,t]] + (C z_lds[d])[t],
                                 sigma2_ev[z_ev[d,t]] + R[t,t] )
    """
    nonev_mean = _nonev_mean(state, params)              # (D, T)
    nonev_var  = params.lds.diag_R()                     # (T,)
    var_dt     = params.sigma2_ev[state.z_ev] + nonev_var[None, :]
    mean_dt    = state.theta[state.z_ev] + nonev_mean
    ll = -0.5 * (np.log(2 * np.pi * var_dt) + (x - mean_dt) ** 2 / var_dt)
    return float(ll.sum())


def compute_loglik_c0(state: GibbsState, x: np.ndarray, params: ModelParams) -> float:
    """log p(x | C=0, z_lds, theta, params) = log p(x | z_ev≡off, z_lds, theta, …):

        Σ_{d,t} log N( x[d,t] ; theta[0] + (C z_lds[d])[t],
                                 sigma2_ev[off] + R[t,t] )

    theta[0] = 0 in our convention, but kept in for generality.
    """
    nonev_mean = _nonev_mean(state, params)              # (D, T)
    nonev_var  = params.lds.diag_R()                     # (T,)
    var_t      = params.sigma2_ev[0] + nonev_var
    mean_dt    = params.mu_theta[0] + nonev_mean         # mu_theta[0] = 0 by convention
    residual   = x - mean_dt
    ll = -0.5 * (np.log(2 * np.pi * var_t[None, :])
                 + residual ** 2 / var_t[None, :])
    return float(ll.sum())


def _nonev_mean(state: GibbsState, params: ModelParams) -> np.ndarray:
    """Per-cell Non-EV mean offset under the current state: (C z_lds[d])[t].

    For the current identity-emission C=I, this is just z_lds. The full matmul
    form is kept so dropping C=I later (lower-dim latent) needs no changes here.
    """
    return state.z_lds @ params.lds.C.T


# ===========================================================================
# Section 3 — Gibbs blocks
# ===========================================================================

# --- Block 1: C (and z_ev | C) — collapsed sampler ----------------------------

def gibbs_block_c_collapsed(
    state:  GibbsState,
    x:      np.ndarray,
    params: ModelParams,
    log_pi: np.ndarray,
    log_P:  np.ndarray,
    rng,
) -> None:
    """Sample C from its exact marginal posterior over z_ev, then z_ev | C.

    Marginal posterior:
        log p(C=1 | x, z_lds, theta, …)  ∝  log p_C + log p(x | C=1, …)
        log p(C=0 | x, z_lds, theta, …)  ∝  log(1 - p_C) + log p(x | C=0, …)

    The C=1 marginal comes from the HMM forward pass (marginalises z_ev).
    The C=0 marginal is the plain Gaussian likelihood with z_ev≡off.

    Mutates state.{c, z_ev, log_f, log_Z1, log_Z0}.
    """
    nonev_mean = _nonev_mean(state, params)
    nonev_var  = params.lds.diag_R()

    # Forward pass under C=1 — gives log p(x | C=1, …) and the filter messages
    log_f, log_Z1 = ev.hmm_forward(x, state.theta, nonev_mean, nonev_var,
                                    params, log_pi, log_P)
    log_Z0        = compute_loglik_c0(state, x, params)     # log p(x | C=0, …)

    # Bernoulli draw on softmax of log-weights
    log_w1 = np.log(params.p_C + 1e-300)     + log_Z1
    log_w0 = np.log(1 - params.p_C + 1e-300) + log_Z0
    p_c1   = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))
    state.c = int(rng.random() < p_c1)

    # z_ev | C — backward-sample if C=1, otherwise pin to off
    state.z_ev = (ev.hmm_backward_sample(log_f, params, rng) if state.c == 1
                  else np.zeros_like(state.z_ev))

    # Cache for the loglik trace
    state.log_f  = log_f
    state.log_Z1 = log_Z1
    state.log_Z0 = log_Z0


# --- Block 1 (legacy): mixture z_ev step + logistic C step --------------------

def gibbs_block_z_ev_mixture(
    state:  GibbsState,
    x:      np.ndarray,
    params: ModelParams,
    log_pi: np.ndarray,
    log_P:  np.ndarray,
    rng,
) -> None:
    """Legacy mixture-Gibbs z_ev step: propose FFBS draw, accept vs z≡off.

    Acceptance probability comes from the same softmax as the collapsed
    sampler — the difference is that here we always overwrite z_ev (either
    with the FFBS draw or with all-off), rather than first deciding C.

    Mutates state.{z_ev, log_Z1, log_Z0}.
    """
    nonev_mean = _nonev_mean(state, params)
    nonev_var  = params.lds.diag_R()

    z_candidate, log_Z1 = ev.ffbs(x, state.theta, nonev_mean, nonev_var,
                                   params, log_pi, log_P, rng)
    log_Z0 = compute_loglik_c0(state, x, params)

    log_w1   = np.log(params.p_C + 1e-300)     + log_Z1
    log_w0   = np.log(1 - params.p_C + 1e-300) + log_Z0
    p_c1_eff = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))

    state.z_ev = (z_candidate if rng.random() < np.clip(p_c1_eff, 0.0, 1.0)
                  else np.zeros_like(state.z_ev))
    state.log_Z1 = log_Z1
    state.log_Z0 = log_Z0


def gibbs_block_c_logistic(
    state: GibbsState,
    rng,
    *,
    c_logistic_model=None,
) -> None:
    """Legacy C-step: P(C=1) from a logistic regression on per-day z_ev
    transition rates.  If no model is supplied, uses a hard threshold at 1.0.

    Mutates state.c.
    """
    D = state.z_ev.shape[0]
    transitions_per_day = float((np.diff(state.z_ev, axis=1) != 0).sum() / D)
    if c_logistic_model is not None:
        p_c1 = float(c_logistic_model.predict_proba([[transitions_per_day]])[0, 1])
    else:
        p_c1 = float(transitions_per_day > 1.0)
    state.c = int(rng.random() < p_c1)


# --- Block 2: Θ_k  (truncated-Gaussian conjugate, per state k) ----------------

def gibbs_block_theta(
    state:  GibbsState,
    x:      np.ndarray,
    params: ModelParams,
    rng,
) -> None:
    """Sample Θ_k for k ∈ {low, high} from its conditional truncated Normal.

    Mutates state.theta[1] and state.theta[2].
    """
    nonev_mean = _nonev_mean(state, params)
    nonev_var  = params.lds.diag_R()
    for k in (1, 2):
        state.theta[k] = ev.sample_theta_k(
            x, state.z_ev, nonev_mean, nonev_var, params, k, rng,
        )


# --- Block 3: z_lds  (Kalman FFBS on EV-residuals) ----------------------------

def gibbs_block_z_lds(
    state:  GibbsState,
    x:      np.ndarray,
    params: ModelParams,
    rng,
) -> None:
    """Sample z^LDS_{1:D} from its conditional posterior given (z_ev, theta).

    Residual x - theta[z_ev] is treated as a noisy LDS observation with
    per-cell extra noise sigma2_ev[z_ev[d,t]] added to diag(R).

    Mutates state.z_lds.
    """
    state.z_lds = non_ev_lds.sample_z_lds(
        x, state.z_ev, state.theta, params.sigma2_ev, params.lds, rng,
    )


# ===========================================================================
# Section 4 — Storage helpers
# ===========================================================================

@dataclass
class _Buffers:
    """All arrays we accumulate or trace across the Gibbs chain.

    Per-iter LDS latent z^LDS is too large to store fully; instead we
    accumulate the posterior mean (`z_lds_running_sum / S`) over retained
    iters and keep the last draw for visualization.
    """
    # Post-burn-in accumulators
    z_counts: np.ndarray                # (D, T, K)
    z_lds_running_sum: np.ndarray       # (D, T)   accumulates retained z_lds samples

    # Post-burn-in samples
    theta_samples:                 np.ndarray   # (S, K)
    c_samples:                     np.ndarray   # (S,)
    c_from_z_samples:              np.ndarray   # (S,)
    z_transitions_per_day_samples: np.ndarray   # (S,)

    # Full traces (length n_total = S_burn + S)
    theta_trace:     np.ndarray | None
    state_occ_trace: np.ndarray | None
    loglik_trace:    np.ndarray | None
    log_Z1_trace:    np.ndarray | None
    log_Z0_trace:    np.ndarray | None


def _allocate_buffers(
    D: int, S: int, n_total: int, record_traces: bool, record_marginals: bool,
) -> _Buffers:
    return _Buffers(
        z_counts          = np.zeros((D, T, K), dtype=np.float64) if record_marginals else None,
        z_lds_running_sum = np.zeros((D, T),    dtype=np.float64),
        theta_samples                 = np.zeros((S, K), dtype=np.float64),
        c_samples                     = np.zeros(S,      dtype=np.int8),
        c_from_z_samples              = np.zeros(S,      dtype=np.int8),
        z_transitions_per_day_samples = np.zeros(S,      dtype=np.float64),
        theta_trace     = np.zeros((n_total, K), dtype=np.float64) if record_traces else None,
        state_occ_trace = np.zeros((n_total, K), dtype=np.float64) if record_traces else None,
        loglik_trace    = np.zeros(n_total,      dtype=np.float64) if record_traces else None,
        log_Z1_trace    = np.zeros(n_total,      dtype=np.float64) if record_traces else None,
        log_Z0_trace    = np.zeros(n_total,      dtype=np.float64) if record_traces else None,
    )


def _record_iter_trace(buf: _Buffers, state: GibbsState, x: np.ndarray,
                       params: ModelParams, it: int) -> None:
    """Append this iter's state to the per-iter traces (if enabled)."""
    if buf.theta_trace is None:
        return
    buf.theta_trace[it]     = state.theta
    buf.state_occ_trace[it] = [(state.z_ev == k).mean() for k in range(K)]
    buf.loglik_trace[it]    = compute_loglik(state, x, params)
    buf.log_Z1_trace[it]    = state.log_Z1
    buf.log_Z0_trace[it]    = state.log_Z0


def _accumulate_post_burnin(buf: _Buffers, state: GibbsState, s_idx: int, D: int) -> None:
    """Add retained-iter info to the post-burn-in summaries."""
    buf.theta_samples[s_idx] = state.theta
    buf.c_samples[s_idx]     = state.c
    if buf.z_counts is not None:
        for k in range(K):
            buf.z_counts[:, :, k] += (state.z_ev == k)
    buf.z_lds_running_sum += state.z_lds
    buf.c_from_z_samples[s_idx]              = int(np.any(state.z_ev != 0))
    buf.z_transitions_per_day_samples[s_idx] = float(
        (np.diff(state.z_ev, axis=1) != 0).sum() / D
    )


def _maybe_print_progress(it: int, n_total: int, S_burn: int, state: GibbsState,
                          buf: _Buffers, t_start: float, verbose: bool) -> None:
    """Print a progress line at iters 0..2, the burn-in/keep boundary, and every 100th."""
    if not verbose:
        return
    if not (it < 3 or it == S_burn or (it + 1) % 100 == 0):
        return
    phase   = "burn-in" if it < S_burn else "keep  "
    elapsed = time.time() - t_start
    ll      = buf.loglik_trace[it] if buf.loglik_trace is not None else float("nan")
    print(f"    iter {it+1:4d}/{n_total} [{phase}]  "
          f"C={state.c}  Θ_low={state.theta[1]:.3f}  Θ_high={state.theta[2]:.3f}  "
          f"z_lds∈[{state.z_lds.min():+.2f},{state.z_lds.max():+.2f}]  "
          f"logL={ll:.1f}  ({elapsed:.1f}s)")


def _build_home_inference_result(
    home_id: int, S_burn: int, S: int, buf: _Buffers, state: GibbsState,
) -> HomeInference:
    z_marginals = buf.z_counts / S if buf.z_counts is not None else None
    z_hat       = (np.argmax(z_marginals, axis=2) if z_marginals is not None
                   else np.zeros_like(state.z_ev))
    c_hat_prob  = float(buf.c_samples.mean())

    return HomeInference(
        home_id                       = home_id,
        C_hat                         = int(c_hat_prob >= 0.5),
        z_hat                         = z_hat,
        z_marginals                   = z_marginals,
        theta_samples                 = buf.theta_samples,
        z_lds_mean                    = buf.z_lds_running_sum / S,
        z_lds_last                    = state.z_lds.copy(),
        c_samples                     = buf.c_samples,
        c_from_z_samples              = buf.c_from_z_samples,
        z_transitions_per_day_samples = buf.z_transitions_per_day_samples,
        theta_trace                   = buf.theta_trace,
        state_occ_trace               = buf.state_occ_trace,
        loglik_trace                  = buf.loglik_trace,
        log_Z1_trace                  = buf.log_Z1_trace,
        log_Z0_trace                  = buf.log_Z0_trace,
        S_burn                        = S_burn,
    )


# ===========================================================================
# Section 5 — Per-home samplers
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

    Per-iteration block sequence — each call is a small named function below:

        gibbs_block_c_collapsed   →   samples (C, z_ev) jointly, marginalising z_ev for C
        gibbs_block_theta         →   samples Θ_k truncated-Normal, per state k
        gibbs_block_z_lds         →   samples z^LDS via Kalman FFBS on residuals
    """
    if rng is None:
        rng = np.random.default_rng(0)
    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"
    if verbose:
        print(f"  [home {home_id}] D={D} → collapsed Gibbs "
              f"({S_burn} burn-in + {S} retained)")

    # ── set-up ───────────────────────────────────────────────────────────
    state  = _initial_state(D, params)
    buf    = _allocate_buffers(D, S, S_burn + S, record_traces, record_marginals=True)
    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    # ── main loop ────────────────────────────────────────────────────────
    t_start = time.time()
    for it in range(S_burn + S):
        gibbs_block_c_collapsed(state, home_x, params, log_pi, log_P, rng)  # Block 1: (C, z_ev)
        gibbs_block_theta      (state, home_x, params,                rng)  # Block 2: Θ_k
        gibbs_block_z_lds      (state, home_x, params,                rng)  # Block 3: z^LDS

        _record_iter_trace(buf, state, home_x, params, it)
        if it >= S_burn:
            _accumulate_post_burnin(buf, state, it - S_burn, D)
        _maybe_print_progress(it, S_burn + S, S_burn, state, buf, t_start, verbose)

    if verbose:
        _print_home_summary(home_id, time.time() - t_start, buf, state, S)
    return _build_home_inference_result(home_id, S_burn, S, buf, state)


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
    """Legacy mixture-Gibbs sampler for one home.

    Per-iteration block sequence:

        gibbs_block_z_ev_mixture  →   propose FFBS draw vs all-off, sample z_ev
        gibbs_block_c_logistic    →   resample C from logistic-on-transitions heuristic
        gibbs_block_theta         →   samples Θ_k truncated-Normal, per state k
        gibbs_block_z_lds         →   samples z^LDS via Kalman FFBS on residuals
    """
    if rng is None:
        rng = np.random.default_rng(0)
    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"
    if verbose:
        print(f"  [home {home_id}] D={D} → mixture Gibbs "
              f"({S_burn} burn-in + {S} retained)  initial_c={initial_c}")

    # ── set-up (with optional warm starts on c, z_ev) ────────────────────
    state = _initial_state(D, params)
    state.c    = int(initial_c)
    state.z_ev = initial_z.copy() if initial_z is not None else state.z_ev
    buf    = _allocate_buffers(D, S, S_burn + S, record_traces, record_marginals=True)
    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    # ── main loop ────────────────────────────────────────────────────────
    t_start = time.time()
    for it in range(S_burn + S):
        gibbs_block_z_ev_mixture(state, home_x, params, log_pi, log_P, rng)
        gibbs_block_c_logistic  (state, rng, c_logistic_model=c_logistic_model)
        gibbs_block_theta       (state, home_x, params,                rng)
        gibbs_block_z_lds       (state, home_x, params,                rng)

        _record_iter_trace(buf, state, home_x, params, it)
        if it >= S_burn:
            _accumulate_post_burnin(buf, state, it - S_burn, D)
        _maybe_print_progress(it, S_burn + S, S_burn, state, buf, t_start, verbose)

    if verbose:
        _print_home_summary(home_id, time.time() - t_start, buf, state, S)
    return _build_home_inference_result(home_id, S_burn, S, buf, state)


def _print_home_summary(home_id: int, elapsed: float, buf: _Buffers,
                        state: GibbsState, S: int) -> None:
    z_marginals = buf.z_counts / S
    c_hat_prob  = float(buf.c_samples.mean())
    frac        = z_marginals.mean(axis=(0, 1))
    z_lds_mean  = buf.z_lds_running_sum / S
    print(f"\n  [home {home_id}] done in {elapsed:.1f}s")
    print(f"    P̂(C=1) from chain : {c_hat_prob:.4f}  (hard={int(c_hat_prob >= 0.5)})")
    print(f"    z_ev freq : off={frac[0]:.3f}  low={frac[1]:.3f}  high={frac[2]:.3f}")
    print(f"    z_lds posterior mean: min={z_lds_mean.min():+.3f} "
          f"median={np.median(z_lds_mean):+.3f} max={z_lds_mean.max():+.3f}")
    for k in (1, 2):
        print(f"    Θ[{STATE_NAMES[k]:>4}] : "
              f"mean={buf.theta_samples[:,k].mean():.3f}  "
              f"std={buf.theta_samples[:,k].std():.4f}")


# ===========================================================================
# Section 6 — Dataset-level drivers
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
    """Run infer_home_collapsed on every home in df."""
    return _infer_all_impl(df, params, S_burn=S_burn, S=S, seed=seed, verbose=verbose,
                           per_home_fn=infer_home_collapsed,
                           per_home_kwargs={},
                           banner="INFERENCE: collapsed Gibbs over all homes")


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
    """Run mixture-Gibbs infer_home on every home in df."""
    return _infer_all_impl(
        df, params, S_burn=S_burn, S=S, seed=seed, verbose=verbose,
        per_home_fn=infer_home,
        per_home_kwargs={
            "initial_c_dict": initial_c_dict or {},
            "initial_z_dict": initial_z_dict or {},
            "c_logistic_model": c_logistic_model,
        },
        banner="INFERENCE: mixture Gibbs over all homes",
        is_legacy=True,
    )


def _infer_all_impl(
    df: pd.DataFrame,
    params: ModelParams,
    *,
    S_burn: int, S: int, seed: int, verbose: bool,
    per_home_fn, per_home_kwargs: dict, banner: str, is_legacy: bool = False,
) -> dict[int, HomeInference]:
    """Shared driver loop: iterate over homes and dispatch to per_home_fn."""
    if verbose:
        print("=" * 60); print(banner); print("=" * 60)
    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    homes = list(sorted_df["home_id"].unique())
    rng   = np.random.default_rng(seed)

    results: dict[int, HomeInference] = {}
    t0 = time.time()
    for i, hid in enumerate(homes):
        g = sorted_df[sorted_df["home_id"] == hid]
        D = len(g) // T
        x = g["total_load"].to_numpy().reshape(D, T).astype(np.float64)

        # Per-home call site: assemble kwargs (legacy sampler takes warm-start dicts)
        call_kwargs = dict(S_burn=S_burn, S=S, rng=rng, home_id=int(hid), verbose=verbose)
        if is_legacy:
            call_kwargs["initial_c"] = int(per_home_kwargs["initial_c_dict"].get(int(hid), 1))
            call_kwargs["initial_z"] = per_home_kwargs["initial_z_dict"].get(int(hid), None)
            call_kwargs["c_logistic_model"] = per_home_kwargs["c_logistic_model"]

        if verbose:
            true_c = int(g["has_ev"].iloc[0]) if "has_ev" in g.columns else "?"
            tag    = f"  init_c={call_kwargs.get('initial_c')}" if is_legacy else ""
            print(f"\n[{i+1}/{len(homes)}] home {hid}  D={D}  true_c={true_c}{tag}")

        results[int(hid)] = per_home_fn(x, params, **call_kwargs)

    if verbose:
        print(f"\nAll homes done in {time.time() - t0:.1f}s")
    return results


# ===========================================================================
# Section 7 — Heuristic adapters
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
