"""
Generative graphical model for residential power: fit + Gibbs inference.

Notation follows specs/graphical_model.tex. See specs/fit.md and
specs/inference.md for derivations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T = 24 * 60 // 15        # 15-min intervals per day
K = 3                 # number of states: 0=off, 1=low, 2=high
STATE_NAMES = ["off", "low", "high"]
LAPLACE = 1e-3        # smoothing for HMM transition counts
SIGMA_EV_OFF = 1e-3   # floor for off-state EV emission std
EM_TOL = 1e-6
EM_MAX_ITERS = 100
THETA_VAR_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """All globally point-estimated parameters."""

    # EV state
    p_C: float                  # EV prevalence
    pi_z: np.ndarray            # (K, ) initial EV state probabilities for time 0
    P_z: np.ndarray             # (K, K) EV transition probabilities

    # EV charging magnitudes
    mu_theta: np.ndarray          # (K,) Per-state EV charging mean.First term is fixed to 0.
    sigma2_theta: np.ndarray      # (K,) Per-state EV mean charging variance. First term is fixed to 0.
    sigma2_ev: np.ndarray         # (K,) Per-state, observation EV charging variance.

    # Background
    rho: np.ndarray               # (T,) background shape with unit norm
    mu_alpha: float               # mean of per-home background loads
    sigma2_alpha: float           # variance of per-home background loads
    sigma2_nonev: np.ndarray      # (T,) per-time heteroscedastic variance of non-EV loads

    K: int = K
    T: int = T

    def summary(self) -> str:
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
                f"  Theta[{name:>4}]: mu={self.mu_theta[k]:.4f}, "
                f"sigma_Theta={np.sqrt(self.sigma2_theta[k]):.4f}, "
                f"sigma^EV={np.sqrt(self.sigma2_ev[k]):.4f}"
            )
        
        lines.append("\nNon-EV")
        lines += [
            f"  rho                 (||.||_2 = {np.linalg.norm(self.rho):.4f}, "
            f"min={self.rho.min():+.3f}, max={self.rho.max():+.3f}, mean={self.rho.mean():+.3f})",
            f"  mu_alpha            = {self.mu_alpha:.4f}",
            f"  sigma_alpha         = {np.sqrt(self.sigma2_alpha):.4f}",
            f"  sigma^NonEV_t       (min={np.sqrt(self.sigma2_nonev.min()):.3f}, "
            f"median={np.sqrt(np.median(self.sigma2_nonev)):.3f}, "
            f"max={np.sqrt(self.sigma2_nonev.max()):.3f})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference container
# ---------------------------------------------------------------------------

@dataclass
class HomeInference:
    home_id: int
    C_hat: int
    z_hat: np.ndarray                            # (D, T) MAP states

    # Post-burn-in posterior summaries
    z_marginals: np.ndarray | None = None        # (D, T, K)
    alpha_samples: np.ndarray | None = None      # (S,)
    theta_samples: np.ndarray | None = None      # (S, K)

    # Per-sample C drawn from the Gibbs chain  (primary C estimate)
    c_samples: np.ndarray | None = None                       # (S,) int {0, 1}

    # Per-sample C indicator: 1 if any z[d,t] != off in that sample (derived)
    c_from_z_samples: np.ndarray | None = None                # (S,) binary

    # Per-sample within-day z transition rate (for heuristic-based C estimation)
    z_transitions_per_day_samples: np.ndarray | None = None   # (S,) float

    # Full iteration traces (burn-in + retained), for convergence diagnostics
    alpha_trace: np.ndarray | None = None        # (S_burn + S,)
    theta_trace: np.ndarray | None = None        # (S_burn + S, K)
    state_occ_trace: np.ndarray | None = None    # (S_burn + S, K) fraction in each state
    loglik_trace: np.ndarray | None = None       # (S_burn + S,) complete-data log-likelihood

    S_burn: int = 0                              # number of burn-in iterations


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit(train_df: pd.DataFrame, *, verbose: bool = True) -> ModelParams:
    """Fit all global parameters from a training dataframe.

    Required columns: home_id, day, time_index, total_load, ev_load,
    non_ev_load, charge_state, has_ev.
    """
    if verbose:
        print("=" * 60)
        print("FIT: graphical model")
        print("=" * 60)

    # Group per (home_id, day, time_index) — assume already complete days.
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
    # Step 1 — p_C
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print("\n[Step 1] EV prevalence p_C")
    p_C = sorted_df.groupby("home_id")["has_ev"].first().mean()
    if verbose:
        print(f"  p_C = {p_C:.4f} ({int(p_C * N)}/{N} homes have EV)")
        print(f"  Step 1 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Pre-shape per-home arrays for downstream steps
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Pre-shape] Building per-home (D, T) arrays")
    t0 = time.time()
    home_arrays = _build_home_arrays(sorted_df, homes)
    if verbose:
        ds = [a["D"] for a in home_arrays.values()]
        print(f"  D^(n) range: min={min(ds)}, median={int(np.median(ds))}, max={max(ds)}")
        print(f"  Pre-shape done in {time.time() - t0:.3f}s")

    ev_homes = [hid for hid in homes if home_arrays[hid]["has_ev"]]
    N_EV = len(ev_homes)

    # ------------------------------------------------------------------
    # Step 2 — HMM (EV homes only)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 2] HMM parameters from {N_EV} EV homes")
    pi_z, P_z = _fit_hmm(home_arrays, ev_homes, verbose=verbose)
    if verbose:
        print(f"  Step 2 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 3 — Background block (all homes)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 3] Background block from all {N} homes")
    rho, mu_alpha, sigma2_alpha, sigma2_nonev = _fit_background(home_arrays, homes, verbose=verbose)
    if verbose:
        print(f"  Step 3 done in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 4 — EV charging magnitudes (EM, EV homes only)
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
        print(f"\n[Step 4] EV charging magnitudes (EM) from {N_EV} EV homes")
    mu_theta, sigma2_theta, sigma2_ev = _fit_charging_em(home_arrays, ev_homes, verbose=verbose)
    if verbose:
        print(f"  Step 4 done in {time.time() - t0:.3f}s")

    params = ModelParams(
        p_C=float(p_C),
        pi_z=pi_z,
        P_z=P_z,
        rho=rho,
        mu_alpha=float(mu_alpha),
        sigma2_alpha=float(sigma2_alpha),
        sigma2_nonev=sigma2_nonev,
        mu_theta=mu_theta,
        sigma2_theta=sigma2_theta,
        sigma2_ev=sigma2_ev,
    )

    if verbose:
        print("\n" + params.summary())

    return params


# ---------------------------------------------------------------------------
# Helpers: per-home array assembly
# ---------------------------------------------------------------------------

def _build_home_arrays(sorted_df: pd.DataFrame, homes: Iterable[int]) -> dict:
    """Reshape long dataframe to per-home (D, T) arrays for total/ev/non-ev/state."""
    out = {}
    for hid, g in sorted_df.groupby("home_id", sort=False):
        days = g["day"].to_numpy()
        # group rows by day → (D, T) reshape
        D = len(np.unique(days))
        if len(g) != D * T:
            raise ValueError(
                f"home {hid}: expected D*T={D*T} rows, got {len(g)} (incomplete days?)"
            )
        out[int(hid)] = {
            "has_ev": bool(g["has_ev"].iloc[0]),
            "city": g["city"].iloc[0],
            "D": D,
            "x": g["total_load"].to_numpy().reshape(D, T).astype(np.float64),
            "x_ev": g["ev_load"].to_numpy().reshape(D, T).astype(np.float64),
            "x_nev": g["non_ev_load"].to_numpy().reshape(D, T).astype(np.float64),
            "z": g["charge_state"].to_numpy().reshape(D, T).astype(np.int64),
        }
    return out


# ---------------------------------------------------------------------------
# Step 2 — HMM
# ---------------------------------------------------------------------------

def _fit_hmm(home_arrays: dict, ev_homes: list[int], *, verbose: bool):
    pi_counts = np.zeros(K, dtype=np.float64)
    trans_counts = np.zeros((K, K), dtype=np.float64)

    for hid in ev_homes:
        z = home_arrays[hid]["z"]              # (D, T)
        # Initial: state at t=0 across all days
        starts = z[:, 0]
        for k in range(K):
            pi_counts[k] += int(np.sum(starts == k))
        # Transitions: within-day pairs (t-1, t) for t in 1..T-1
        prev = z[:, :-1].ravel()
        nxt = z[:, 1:].ravel()
        for k in range(K):
            mask = (prev == k)
            if not mask.any():
                continue
            for kp in range(K):
                trans_counts[k, kp] += int(np.sum(nxt[mask] == kp))

    pi_z = pi_counts / pi_counts.sum()

    # Laplace-smoothed row-normalized transitions
    smoothed = trans_counts + LAPLACE
    P_z = smoothed / smoothed.sum(axis=1, keepdims=True)

    if verbose:
        print(f"  raw initial counts: {pi_counts.astype(int).tolist()}")
        print(f"  pi_z = {np.array2string(pi_z, precision=4)}")
        print(f"  raw transition counts (rows from→cols to):")
        for k in range(K):
            print(f"    {STATE_NAMES[k]:>4}: {trans_counts[k].astype(int).tolist()}")
        print(f"  P_z (smoothed, row-normalized):")
        for k in range(K):
            print(f"    {STATE_NAMES[k]:>4}: {np.array2string(P_z[k], precision=4)}")

    return pi_z, P_z


# ---------------------------------------------------------------------------
# Step 3 — Background block
# ---------------------------------------------------------------------------

def _fit_background(home_arrays: dict, homes: list[int], *, verbose: bool):
    home_ids = list(homes)
    N = len(home_ids)

    # 3a. Per-home day-mean profile β̂^(n)_t  -> B (N, T)
    B = np.stack([home_arrays[hid]["x_nev"].mean(axis=0) for hid in home_ids])  # (N, T)
    Ds = np.array([home_arrays[hid]["D"] for hid in home_ids])                  # (N,)

    # 3b. SVD of B → ρ (right singular vector)
    U, S_vals, Vt = np.linalg.svd(B, full_matrices=False)
    rho = Vt[0].copy()  # (T,)
    # 3c. plug-in α̂^(n) = β̂^(n) · ρ
    alpha_hat = B @ rho                                                          # (N,)
    # Sign convention: median(α̂) > 0
    if np.median(alpha_hat) < 0:
        rho = -rho
        alpha_hat = -alpha_hat
    mu_alpha = alpha_hat.mean()

    if verbose:
        print(f"  SVD: top-3 singular values = {np.array2string(S_vals[:3], precision=2)}")
        print(f"       singular value ratio sv1/sv2 = {S_vals[0] / S_vals[1]:.2f} "
              f"(higher → more rank-1 dominant)")
        print(f"  rho: ||.||_2 = {np.linalg.norm(rho):.4f}, "
              f"min={rho.min():+.4f}, max={rho.max():+.4f}")
        print(f"  alpha_hat (per-home scale): "
              f"min={alpha_hat.min():.3f}, median={np.median(alpha_hat):.3f}, "
              f"max={alpha_hat.max():.3f}")
        print(f"  mu_alpha = {mu_alpha:.4f}")

    # 3d. (σ^NonEV_t)² from individual obs
    # Vectorize: for each home, compute squared residuals, accumulate per t
    sum_resid_sq = np.zeros(T, dtype=np.float64)
    total_obs_per_t = 0  # = sum of D^(n)
    for hid, ah in zip(home_ids, alpha_hat):
        x_nev = home_arrays[hid]["x_nev"]                                       # (D, T)
        residuals = x_nev - ah * rho[None, :]                                   # (D, T)
        sum_resid_sq += (residuals ** 2).sum(axis=0)                            # (T,)
        total_obs_per_t += home_arrays[hid]["D"]
    sigma2_nonev = sum_resid_sq / total_obs_per_t                                # (T,)

    if verbose:
        sig = np.sqrt(sigma2_nonev)
        print(f"  sigma^NonEV_t: min={sig.min():.3f}, median={np.median(sig):.3f}, "
              f"max={sig.max():.3f}")
        print(f"    (these should be on the kW scale, comparable to background load magnitudes)")

    # 3e. σ_α² with bias correction
    var_alpha_raw = float(np.var(alpha_hat, ddof=1))
    # correction = (1/N) Σ_n (1/D^(n)) Σ_t ρ_t² (σ^NonEV_t)²
    bias_per_home = (1.0 / Ds) * np.sum(rho ** 2 * sigma2_nonev)                # (N,)
    bias = bias_per_home.mean()
    sigma2_alpha = max(0.0, var_alpha_raw - bias)

    if verbose:
        print(f"  Var_n(alpha_hat) = {var_alpha_raw:.5f}")
        print(f"  bias correction  = {bias:.5f} (subtracted)")
        print(f"  sigma2_alpha     = {sigma2_alpha:.5f}  (sigma_alpha = {np.sqrt(sigma2_alpha):.4f})")

    return rho, mu_alpha, sigma2_alpha, sigma2_nonev


# ---------------------------------------------------------------------------
# Step 4 — EV charging magnitudes (EM)
# ---------------------------------------------------------------------------

def _fit_charging_em(home_arrays: dict, ev_homes: list[int], *, verbose: bool):
    """Run EM separately for k ∈ {1, 2}; off state is fixed."""

    mu_theta = np.zeros(K, dtype=np.float64)
    sigma2_theta = np.zeros(K, dtype=np.float64)
    sigma2_ev = np.zeros(K, dtype=np.float64)
    sigma2_ev[0] = SIGMA_EV_OFF ** 2          # floor for off

    for k in (1, 2):
        # Sufficient statistics per home for state k
        n_per_home = []   # (N_EV,)
        S_y = []
        SS_y = []
        for hid in ev_homes:
            z = home_arrays[hid]["z"]
            x_ev = home_arrays[hid]["x_ev"]
            mask = (z == k)
            n_per_home.append(int(mask.sum()))
            obs = x_ev[mask]
            S_y.append(float(obs.sum()))
            SS_y.append(float((obs ** 2).sum()))

        n_per_home = np.array(n_per_home, dtype=np.float64)
        S_y = np.array(S_y, dtype=np.float64)
        SS_y = np.array(SS_y, dtype=np.float64)
        N_EV = len(ev_homes)

        if verbose:
            print(f"\n  --- State k={k} ({STATE_NAMES[k]}) ---")
            print(f"  n^(n)_k per home: min={int(n_per_home.min())}, "
                  f"median={int(np.median(n_per_home))}, max={int(n_per_home.max())}, "
                  f"sum={int(n_per_home.sum())}")
            if n_per_home.min() == 0:
                zero_homes = [ev_homes[i] for i, n in enumerate(n_per_home) if n == 0]
                print(f"  WARNING: {len(zero_homes)} home(s) never visit state {k}: {zero_homes}")

        if n_per_home.sum() == 0:
            print(f"  No observations in state {k}; skipping (using prior).")
            mu_theta[k] = 0.0
            sigma2_theta[k] = 0.0
            sigma2_ev[k] = 0.01
            continue

        # Avoid divide-by-zero for any home with n=0 (skip those in MoM init)
        active = n_per_home > 0
        theta_hat = np.where(active, S_y / np.maximum(n_per_home, 1), 0.0)

        # ANOVA initialization
        N_k = n_per_home.sum()
        SS_within = SS_y[active].sum() - (n_per_home[active] * theta_hat[active] ** 2).sum()
        denom = N_k - active.sum()
        sigma2_ev_k = SS_within / max(denom, 1.0)
        var_theta_hat = float(np.var(theta_hat[active], ddof=1)) if active.sum() > 1 else 0.0
        mean_inv_n = float(np.mean(1.0 / n_per_home[active]))
        sigma2_theta_k = max(0.0, var_theta_hat - sigma2_ev_k * mean_inv_n)
        mu_theta_k = float(theta_hat[active].mean())

        if verbose:
            print(f"  ANOVA init: mu={mu_theta_k:.4f}, "
                  f"sigma_Theta={np.sqrt(sigma2_theta_k):.4f}, "
                  f"sigma^EV={np.sqrt(sigma2_ev_k):.4f}")

        # EM iterations
        prev_loglik = -np.inf
        for it in range(EM_MAX_ITERS):
            sig2_theta = max(sigma2_theta_k, THETA_VAR_FLOOR)

            # E-step
            prec = 1.0 / sig2_theta + n_per_home / sigma2_ev_k                  # (N_EV,)
            E_theta = (mu_theta_k / sig2_theta + S_y / sigma2_ev_k) / prec      # (N_EV,)
            Var_theta = 1.0 / prec
            E_theta2 = Var_theta + E_theta ** 2

            # Marginal log-lik (monotone non-decreasing under EM)
            loglik = _charging_loglik(n_per_home, theta_hat, S_y, SS_y,
                                       mu_theta_k, sig2_theta, sigma2_ev_k, active)

            # M-step
            mu_theta_k = float(np.mean(E_theta))
            sigma2_theta_k = float(np.mean(Var_theta + (E_theta - mu_theta_k) ** 2))
            sigma2_ev_k = float(
                np.sum(SS_y - 2 * S_y * E_theta + n_per_home * E_theta2) / N_k
            )

            delta = loglik - prev_loglik
            if verbose and (it < 5 or it % 10 == 0 or abs(delta) < EM_TOL):
                print(f"    iter {it:3d}: logL={loglik:.4f}  Δ={delta:+.2e}  "
                      f"mu={mu_theta_k:.4f}  σΘ={np.sqrt(max(sigma2_theta_k,0)):.4f}  "
                      f"σEV={np.sqrt(sigma2_ev_k):.4f}")

            if abs(delta) < EM_TOL and it > 0:
                if verbose:
                    print(f"  EM converged at iter {it}")
                break
            prev_loglik = loglik

        mu_theta[k] = mu_theta_k
        sigma2_theta[k] = max(sigma2_theta_k, 0.0)
        sigma2_ev[k] = sigma2_ev_k

    return mu_theta, sigma2_theta, sigma2_ev


def _charging_loglik(n, theta_hat, S_y, SS_y, mu, sig2_theta, sig2_ev, active):
    """Marginal log-likelihood of one-way Gaussian RE model (sum over homes)."""
    n_a = n[active]
    th_a = theta_hat[active]
    SS_a = SS_y[active]
    within_ss = SS_a - n_a * th_a ** 2  # = Σ (y - ȳ)²

    ll = -0.5 * np.sum(
        n_a * np.log(2 * np.pi)
        + (n_a - 1) * np.log(sig2_ev)
        + np.log(sig2_ev + n_a * sig2_theta)
        + within_ss / sig2_ev
        + n_a * (th_a - mu) ** 2 / (sig2_ev + n_a * sig2_theta)
    )
    return float(ll)


# ===========================================================================
# Inference
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
    c_logistic_model=None,   # fitted sklearn LogisticRegression from first_diff_logistic.tune()
) -> HomeInference:
    """Mixture Gibbs inference for one home.  C is sampled in the chain.

    home_x           : (D, T) total load — the ONLY signal used at test time.
    initial_c        : warm-start C value (0 or 1); chain mixes away after burn-in.
    initial_z        : (D, T) warm-start z; defaults to all-off if None.
    c_logistic_model : if provided, C is sampled each iteration from
                       Bernoulli(logistic(transitions_per_day(z))).
                       If None, falls back to a hard threshold at 1 transition/day.

    Why logistic on transitions, not "any z≠off"?
      P(z=all-off | D≈360 days, C=1) ≈ 0.25^360 ≈ 0, so the any-non-off rule
      fires every iteration for all homes, making C≡1 and defeating inference.
      Transitions/day measures *concentrated* charging patterns (EV homes have
      sustained charging blocks → many transitions) vs scattered noise (non-EV
      homes have occasional HMM artefacts → few transitions).

    Gibbs blocks per iteration:
      1. z  | C-marginalised, α, Θ, x  — mixture FFBS (collapsed sampler)
      2. C  | z                         — Bernoulli from logistic(transitions/day)
      3. Θ_k | z, α, x                 — conjugate Gaussian (prior if no state-k obs)
      4. α  | z, Θ, x                  — conjugate Gaussian (always)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"

    if verbose:
        print(f"  [home {home_id}] D={D} → "
              f"mixture Gibbs ({S_burn} burn-in + {S} retained)  "
              f"initial_c={initial_c}")

    # ── initialise ────────────────────────────────────────────────────────────
    alpha = params.mu_alpha
    theta = params.mu_theta.copy()
    z     = initial_z.copy() if initial_z is not None else np.zeros((D, T), dtype=np.int64)
    c     = initial_c

    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    # ── storage ───────────────────────────────────────────────────────────────
    n_total                       = S_burn + S
    z_counts                      = np.zeros((D, T, K), dtype=np.float64)
    alpha_samples                 = np.zeros(S,         dtype=np.float64)
    theta_samples                 = np.zeros((S, K),    dtype=np.float64)
    c_samples                     = np.zeros(S,         dtype=np.int8)
    c_from_z_samples              = np.zeros(S,         dtype=np.int8)
    z_transitions_per_day_samples = np.zeros(S,         dtype=np.float64)

    if record_traces:
        alpha_trace     = np.zeros(n_total,      dtype=np.float64)
        theta_trace     = np.zeros((n_total, K), dtype=np.float64)
        state_occ_trace = np.zeros((n_total, K), dtype=np.float64)
        loglik_trace    = np.zeros(n_total,      dtype=np.float64)
    else:
        alpha_trace = theta_trace = state_occ_trace = loglik_trace = None

    # ── main loop ─────────────────────────────────────────────────────────────
    t_start = time.time()
    s_idx   = -1

    for it in range(n_total):

        # Block 1 — mixture FFBS: sample z marginalising over C ───────────────
        z_candidate, log_Z1 = _ffbs(home_x, theta, alpha, params, log_pi, log_P, rng)

        log_Z0  = _compute_loglik_c0(home_x, alpha, params)
        log_w1  = np.log(params.p_C + 1e-300)     + log_Z1
        log_w0  = np.log(1 - params.p_C + 1e-300) + log_Z0
        p_c1_eff = float(np.exp(log_w1 - float(np.logaddexp(log_w1, log_w0))))

        z = z_candidate if rng.random() < np.clip(p_c1_eff, 0.0, 1.0)             else np.zeros((D, T), dtype=np.int64)

        # Block 2 — sample C | z via transitions/day ─────────────────────────
        # Transitions/day captures sustained charging blocks (EV) vs scattered
        # HMM artefacts (non-EV), avoiding the any-non-off rule that fires for
        # every home since z=all-off is astronomically unlikely for D≈360 days.
        transitions_per_day_now = float((np.diff(z, axis=1) != 0).sum() / D)
        if c_logistic_model is not None:
            p_c1 = float(c_logistic_model.predict_proba([[transitions_per_day_now]])[0, 1])
        else:
            p_c1 = float(transitions_per_day_now > 1.0)   # hard threshold fallback
        c = int(rng.random() < p_c1)

        # Block 3 — sample Θ_k (draws from prior when z=all-off) ─────────────
        for k in (1, 2):
            theta[k] = _sample_theta_k(home_x, z, alpha, params, k, rng)

        # Block 4 — sample α (always) ─────────────────────────────────────────
        alpha = _sample_alpha(home_x, z, theta, params, rng)

        # ── record traces ─────────────────────────────────────────────────────
        if record_traces:
            alpha_trace[it]     = alpha
            theta_trace[it]     = theta
            state_occ_trace[it] = [(z == k).mean() for k in range(K)]
            loglik_trace[it]    = _compute_loglik(home_x, z, theta, alpha, params)

        # ── accumulate post-burn-in ───────────────────────────────────────────
        if it >= S_burn:
            s_idx = it - S_burn
            alpha_samples[s_idx] = alpha
            theta_samples[s_idx] = theta
            c_samples[s_idx]     = c
            for k in range(K):
                z_counts[:, :, k] += (z == k)
            c_from_z_samples[s_idx]              = int(np.any(z != 0))
            z_transitions_per_day_samples[s_idx] = float(
                (np.diff(z, axis=1) != 0).sum() / D
            )

        # ── verbose progress ──────────────────────────────────────────────────
        if verbose and (it < 3 or it == S_burn or (it + 1) % 100 == 0):
            phase   = "burn-in" if it < S_burn else "keep  "
            elapsed = time.time() - t_start
            conv_flag = ""
            if s_idx >= 50:
                window    = alpha_samples[max(0, s_idx - 49): s_idx + 1]
                conv_flag = f"  Δα/σ={abs(window[-1]-window[0])/(window.std()+1e-9)*100:.0f}%"
            ll = loglik_trace[it] if record_traces else float("nan")
            print(f"    iter {it+1:4d}/{n_total} [{phase}]  "
                  f"C={c}  α={alpha:.3f}  Θ_low={theta[1]:.3f}  Θ_high={theta[2]:.3f}  "
                  f"logL={ll:.1f}  ({elapsed:.1f}s){conv_flag}")

    # ── final summaries ───────────────────────────────────────────────────────
    z_marginals = z_counts / S
    z_hat       = np.argmax(z_marginals, axis=2)
    c_hat_prob  = float(c_samples.mean())

    if verbose:
        elapsed = time.time() - t_start
        frac = z_marginals.mean(axis=(0, 1))
        print(f"\n  [home {home_id}] done in {elapsed:.1f}s")
        print(f"    P̂(C=1) from chain : {c_hat_prob:.4f}  (hard={int(c_hat_prob >= 0.5)})")
        print(f"    z freq : off={frac[0]:.3f}  low={frac[1]:.3f}  high={frac[2]:.3f}")
        print(f"    α : mean={alpha_samples.mean():.3f}  std={alpha_samples.std():.4f}")
        for k in (1, 2):
            print(f"    Θ[{STATE_NAMES[k]:>4}] : "
                  f"mean={theta_samples[:,k].mean():.3f}  std={theta_samples[:,k].std():.4f}")

    return HomeInference(
        home_id                       = home_id,
        C_hat                         = int(c_hat_prob >= 0.5),
        z_hat                         = z_hat,
        z_marginals                   = z_marginals,
        alpha_samples                 = alpha_samples,
        theta_samples                 = theta_samples,
        c_samples                     = c_samples,
        c_from_z_samples              = c_from_z_samples,
        z_transitions_per_day_samples = z_transitions_per_day_samples,
        alpha_trace                   = alpha_trace,
        theta_trace                   = theta_trace,
        state_occ_trace               = state_occ_trace,
        loglik_trace                  = loglik_trace,
        S_burn                        = S_burn,
    )

# ---------------------------------------------------------------------------
# Gibbs helpers
# ---------------------------------------------------------------------------

def _compute_loglik(
    x: np.ndarray,      # (D, T)
    z: np.ndarray,      # (D, T) int
    theta: np.ndarray,  # (K,)
    alpha: float,
    params: ModelParams,
) -> float:
    """Complete-data log-likelihood: Σ_{d,t} log N(x[d,t]; θ_{z[d,t]} + α·ρ_t, σ²_{z[d,t],t})."""
    combined_var = params.sigma2_ev[z] + params.sigma2_nonev[None, :]
    mean_dt      = theta[z] + alpha * params.rho[None, :]
    ll           = -0.5 * (np.log(2 * np.pi * combined_var) + (x - mean_dt) ** 2 / combined_var)
    return float(ll.sum())


def _compute_loglik_c0(
    x: np.ndarray,   # (D, T)
    alpha: float,
    params: ModelParams,
) -> float:
    """log p(x | C=0, α) = log p(x | z≡off, α).

    Under C=0, z is always off, so the emission is
    N(x[d,t]; 0 + α·ρ_t, σ²_{EV,off} + σ²_{NonEV,t}).
    """
    combined_var_off = params.sigma2_ev[0] + params.sigma2_nonev          # (T,)
    residual         = x - alpha * params.rho[None, :]                    # (D, T)
    ll = -0.5 * (np.log(2 * np.pi * combined_var_off[None, :])
                 + residual ** 2 / combined_var_off[None, :])
    return float(ll.sum())


# ---------------------------------------------------------------------------
# Gibbs blocks
# ---------------------------------------------------------------------------

def _ffbs(
    x, theta, alpha, params, log_pi, log_P, rng
) -> tuple[np.ndarray, float]:
    """Forward-filter backward-sample, vectorized over days.

    Returns (z, log_Z1) where:
      z       : (D, T) sampled state sequence
      log_Z1  : log p(x | C=1, α, θ) — the marginal likelihood under the EV HMM,
                accumulated as the sum of per-step log-normalisation constants.
                Used by the mixture sampler to weight C=1 vs C=0.
    """
    D = x.shape[0]
    rho          = params.rho
    sigma2_nonev = params.sigma2_nonev
    sigma2_ev    = params.sigma2_ev

    combined_var = sigma2_ev[:, None] + sigma2_nonev[None, :]     # (K, T)
    inv_2var     = 0.5 / combined_var
    log_norm     = -0.5 * np.log(2 * np.pi * combined_var)        # (K, T)

    mean_kt  = theta[:, None] + alpha * rho[None, :]              # (K, T)
    diff     = x[:, :, None] - mean_kt.T[None, :, :]             # (D, T, K)
    log_emit = log_norm.T[None, :, :] - diff ** 2 * inv_2var.T[None, :, :]  # (D, T, K)

    # ── forward pass — accumulate log_Z1 ─────────────────────────────────────
    log_f  = np.empty((D, T, K), dtype=np.float64)
    log_Z1 = 0.0

    unnorm_0      = log_pi[None, :] + log_emit[:, 0, :]           # (D, K)
    lse_0         = logsumexp(unnorm_0, axis=1)                   # (D,)
    log_Z1       += lse_0.sum()
    log_f[:, 0, :] = unnorm_0 - lse_0[:, None]

    for t in range(1, T):
        log_pred    = logsumexp(log_f[:, t-1, :, None] + log_P[None, :, :], axis=1)
        unnorm_t    = log_emit[:, t, :] + log_pred                # (D, K)
        lse_t       = logsumexp(unnorm_t, axis=1)                 # (D,)
        log_Z1     += lse_t.sum()
        log_f[:, t, :] = unnorm_t - lse_t[:, None]

    # ── backward sample ───────────────────────────────────────────────────────
    z     = np.empty((D, T), dtype=np.int64)
    p_T   = np.exp(log_f[:, T-1, :])
    p_T  /= p_T.sum(axis=1, keepdims=True)
    z[:, T-1] = _sample_categorical_rows(p_T, rng)

    P_z = params.P_z
    for t in range(T - 2, -1, -1):
        col = P_z[:, z[:, t+1]].T                                 # (D, K)
        w   = np.exp(log_f[:, t, :]) * col
        w  /= w.sum(axis=1, keepdims=True)
        z[:, t] = _sample_categorical_rows(w, rng)

    return z, log_Z1


def _sample_categorical_rows(probs: np.ndarray, rng) -> np.ndarray:
    """Sample one categorical per row of probs (D, K) → (D,)."""
    cum = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0])[:, None]
    return np.argmax(cum > u, axis=1)


def _sample_theta_k(x, z, alpha, params, k, rng):
    """Conjugate Gaussian sampler for Theta_k."""
    rho = params.rho
    sigma2_nonev = params.sigma2_nonev
    sigma2_ev_k = params.sigma2_ev[k]
    var_t = sigma2_ev_k + sigma2_nonev                                  # (T,)
    inv_var_t = 1.0 / var_t

    mask = (z == k)                                                     # (D, T)
    if not mask.any():
        # draw from prior
        sig2_theta = max(params.sigma2_theta[k], THETA_VAR_FLOOR)
        return rng.normal(params.mu_theta[k], np.sqrt(sig2_theta))

    r = x - alpha * rho[None, :]                                        # (D, T)
    S_inv_var = (mask * inv_var_t[None, :]).sum()
    S_r = (mask * r * inv_var_t[None, :]).sum()

    sig2_prior = max(params.sigma2_theta[k], THETA_VAR_FLOOR)
    prec = 1.0 / sig2_prior + S_inv_var
    m = (params.mu_theta[k] / sig2_prior + S_r) / prec
    return rng.normal(m, np.sqrt(1.0 / prec))


def _sample_alpha(x, z, theta, params, rng):
    """Conjugate Gaussian sampler for alpha."""
    rho = params.rho                                                    # (T,)
    sigma2_nonev = params.sigma2_nonev                                  # (T,)
    sigma2_ev = params.sigma2_ev                                        # (K,)

    # var_dt = sigma2_ev[z[d,t]] + sigma2_nonev[t]  → (D, T)
    var_dt = sigma2_ev[z] + sigma2_nonev[None, :]
    inv_var_dt = 1.0 / var_dt

    # residual at z[d,t]: x[d,t] - theta[z[d,t]]
    r = x - theta[z]                                                    # (D, T)

    # prec_α and m_α
    rho_sq = rho ** 2                                                   # (T,)
    sum_w = (rho_sq[None, :] * inv_var_dt).sum()
    sum_wr = (rho[None, :] * r * inv_var_dt).sum()

    prec = 1.0 / params.sigma2_alpha + sum_w
    m = (params.mu_alpha / params.sigma2_alpha + sum_wr) / prec
    return rng.normal(m, np.sqrt(1.0 / prec))


# ===========================================================================
# Higher-level: run inference over a whole training-style dataframe
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
    """Run mixture Gibbs on every home in df.  C is sampled in the chain.

    initial_c_dict : {home_id: 0|1} warm-start C values (e.g. from heuristic).
                     Defaults to C=1 for all homes if None.
    initial_z_dict : {home_id: (D,T) array} warm-start z values.
                     Defaults to all-off if None.
    """
    if verbose:
        print("=" * 60)
        print("INFERENCE: mixture Gibbs over all homes")
        print("=" * 60)

    sorted_df = df.sort_values(["home_id", "day", "time_index"])
    homes     = list(sorted_df["home_id"].unique())
    rng       = np.random.default_rng(seed)

    results: dict[int, HomeInference] = {}
    t0 = time.time()
    for i, hid in enumerate(homes):
        g   = sorted_df[sorted_df["home_id"] == hid]
        D   = len(g) // T
        x   = g["total_load"].to_numpy().reshape(D, T).astype(np.float64)

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


# ===========================================================================
# Heuristic adapter
# ===========================================================================

def c_prob_from_z_via_heuristic(
    inference: "HomeInference",
    logistic_model,           # fitted sklearn LogisticRegression from first_diff_logistic.tune()
) -> float:
    """Estimate P(C=1) by feeding per-sample z transition counts into the heuristic logistic model.

    For each retained Gibbs sample, the within-day transition rate was recorded.
    We evaluate the pre-trained logistic regression on each and average.

    Flag: the logistic model was trained on transition counts from the heuristic
    state machine on raw load — not on Gibbs z.  Systematic differences in
    transition statistics may cause miscalibration for borderline / non-EV homes.
    For a clear EV home the result should still be near 1.
    """
    rates = inference.z_transitions_per_day_samples   # (S,)
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
# Evaluation
# ===========================================================================

def evaluate(
    df: pd.DataFrame,
    inferences: dict[int, HomeInference],
    c_prob_methods: dict[str, dict[int, float]] | None = None,
    heuristic_states: dict[int, np.ndarray] | None = None,
) -> dict:
    """Build z and C confusion matrices for all inference methods.

    z confusion semantics:
      - Computed per home (normalise each row by that home's true-state count),
        then averaged over homes with equal weight.  NaN for rows with no
        ground-truth examples in that home (e.g. rows 1/2 for non-EV homes).
      - Reported separately for EV homes (C_true=1) and non-EV homes (C_true=0).
      - Both hard (z_hat = argmax marginals) and soft (posterior marginals) versions.
      - If heuristic_states provided: same hard confusion for the heuristic baseline.

    C confusion semantics:
      - For each method in c_prob_methods, one 2×2 hard and one 2×2 soft confusion.
      - Hard: threshold P̂(C=1) at 0.5.  Soft: use P̂ as fractional prediction.
      - Both are row-normalised (i.e. recall-style: fraction of each true class
        going to each predicted class), averaged with equal weight over homes.

    Parameters
    ----------
    c_prob_methods : {method_name: {home_id: P(C=1)}}.
                     For homes where Gibbs was not run (C_hat=0), callers should
                     pre-populate the dict with the heuristic p_hat as fallback.
    heuristic_states : {home_id: flat 1-D array of per-timestep states (0/1/2)}.
    """
    sorted_df = df.sort_values(["home_id", "day", "time_index"])

    # per-home z confusion lists, split by C_true
    ev_hard_cms, ev_soft_cms, ev_heur_cms     = [], [], []
    non_ev_hard_cms, non_ev_soft_cms, non_ev_heur_cms = [], [], []
    ev_home_ids, non_ev_home_ids              = [], []

    for hid, g in sorted_df.groupby("home_id", sort=True):
        hid = int(hid)
        if hid not in inferences:
            continue
        C_true = int(g["has_ev"].iloc[0])
        D = g["day"].nunique()
        z_true = g["charge_state"].to_numpy().reshape(D, T)
        inf    = inferences[hid]

        hard_cm = _per_home_z_confusion_hard(z_true, inf.z_hat)
        soft_cm = (
            _per_home_z_confusion_soft(z_true, inf.z_marginals)
            if inf.z_marginals is not None else None
        )
        heur_cm = None
        if heuristic_states and hid in heuristic_states:
            heur_z = heuristic_states[hid][: D * T].reshape(D, T)
            heur_cm = _per_home_z_confusion_hard(z_true, heur_z)

        if C_true == 1:
            ev_home_ids.append(hid)
            ev_hard_cms.append(hard_cm)
            if soft_cm is not None:
                ev_soft_cms.append(soft_cm)
            if heur_cm is not None:
                ev_heur_cms.append(heur_cm)
        else:
            non_ev_home_ids.append(hid)
            non_ev_hard_cms.append(hard_cm)
            if soft_cm is not None:
                non_ev_soft_cms.append(soft_cm)
            if heur_cm is not None:
                non_ev_heur_cms.append(heur_cm)

    # C confusion for each method
    c_results: dict[str, dict] = {}
    for method_name, c_probs in (c_prob_methods or {}).items():
        c_results[method_name] = _c_confusion_from_probs(sorted_df, inferences, c_probs)

    return {
        # z — EV homes (C_true = 1)
        "ev_home_ids":   ev_home_ids,
        "ev_z_hard":     _nanmean_cms(ev_hard_cms),
        "ev_z_soft":     _nanmean_cms(ev_soft_cms) if ev_soft_cms else None,
        "ev_z_heur":     _nanmean_cms(ev_heur_cms) if ev_heur_cms else None,
        # z — non-EV homes (C_true = 0); rows 1/2 will be NaN
        "non_ev_home_ids": non_ev_home_ids,
        "non_ev_z_hard": _nanmean_cms(non_ev_hard_cms),
        "non_ev_z_soft": _nanmean_cms(non_ev_soft_cms) if non_ev_soft_cms else None,
        "non_ev_z_heur": _nanmean_cms(non_ev_heur_cms) if non_ev_heur_cms else None,
        # C — one dict per method
        "c_results":     c_results,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _per_home_z_confusion_hard(
    z_true: np.ndarray, z_pred: np.ndarray
) -> np.ndarray:
    """Row-normalised hard z confusion for one home. NaN for unobserved true states."""
    cm = np.full((K, K), np.nan)
    for k_true in range(K):
        mask = (z_true == k_true)
        n = int(mask.sum())
        if n == 0:
            continue
        for k_pred in range(K):
            cm[k_true, k_pred] = float((z_pred[mask] == k_pred).sum()) / n
    return cm


def _per_home_z_confusion_soft(
    z_true: np.ndarray, z_marginals: np.ndarray
) -> np.ndarray:
    """Row-normalised soft (expected) z confusion for one home."""
    cm = np.full((K, K), np.nan)
    for k_true in range(K):
        mask = (z_true == k_true)
        n = int(mask.sum())
        if n == 0:
            continue
        cm[k_true] = z_marginals[mask].sum(axis=0) / n
    return cm


def _nanmean_cms(cm_list: list[np.ndarray]) -> np.ndarray | None:
    """Average a list of (K,K) confusion matrices, ignoring NaN entries."""
    if not cm_list:
        return None
    return np.nanmean(np.stack(cm_list, axis=0), axis=0)


def _c_confusion_from_probs(
    sorted_df: pd.DataFrame,
    inferences: dict[int, HomeInference],
    c_probs: dict[int, float],
) -> dict:
    """Hard and soft 2×2 C confusion, row-normalised and averaged over homes."""
    rows = []   # (C_true, C_hard, p_hat)
    for hid, g in sorted_df.groupby("home_id", sort=True):
        hid = int(hid)
        if hid not in inferences or hid not in c_probs:
            continue
        C_true = int(g["has_ev"].iloc[0])
        p_hat  = float(c_probs[hid])
        rows.append((C_true, int(p_hat >= 0.5), p_hat))

    # Build normalised 2×2 hard and soft CMs (equal weight per home)
    hard_cm  = np.zeros((2, 2), dtype=float)
    soft_cm  = np.zeros((2, 2), dtype=float)
    counts   = np.zeros(2, dtype=int)
    for C_true, C_hard, p_hat in rows:
        hard_cm[C_true, C_hard] += 1
        soft_cm[C_true, 0]      += 1 - p_hat
        soft_cm[C_true, 1]      += p_hat
        counts[C_true]           += 1

    with np.errstate(invalid="ignore"):
        hard_cm_norm = np.where(counts[:, None] > 0, hard_cm / counts[:, None], np.nan)
        soft_cm_norm = np.where(counts[:, None] > 0, soft_cm / counts[:, None], np.nan)

    n_correct = sum(1 for C_true, C_hard, _ in rows if C_true == C_hard)
    return {
        "hard_cm":  hard_cm_norm,
        "soft_cm":  soft_cm_norm,
        "accuracy": float(n_correct / max(len(rows), 1)),
        "n_homes":  len(rows),
        "n_ev":     int(counts[1]),
        "n_non_ev": int(counts[0]),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_evaluation(results: dict) -> None:
    """Print evaluation results with clear labels for what is aggregated."""
    SEP = "─" * 64

    def _fmt_row(label: str, row: np.ndarray, n: int | None = None) -> str:
        cells = "  ".join(
            f"{'NaN':>7}" if np.isnan(v) else f"{v:>7.3f}" for v in row
        )
        suffix = f"  (n={n})" if n is not None else ""
        return f"  {label:<8} {cells}{suffix}"

    # ── z confusion ──────────────────────────────────────────────────────────
    for group_label, home_ids, hard_cm, soft_cm, heur_cm in [
        ("EV homes (C_true=1)",
         results["ev_home_ids"],
         results["ev_z_hard"], results["ev_z_soft"], results["ev_z_heur"]),
        ("non-EV homes (C_true=0)",
         results["non_ev_home_ids"],
         results["non_ev_z_hard"], results["non_ev_z_soft"], results["non_ev_z_heur"]),
    ]:
        n_homes = len(home_ids)
        print(f"\n{SEP}")
        print(f"z confusion — {group_label}  (N={n_homes} homes)")
        print(f"  Aggregation: per-home row-normalised CM, then mean over {n_homes} homes")
        print(f"  Rows = true state, columns = predicted state")
        if group_label.startswith("non-EV"):
            print("  Note: rows 'low' and 'high' are NaN (no ground-truth examples)")
        header = f"  {'':8}  {'off':>7}  {'low':>7}  {'high':>7}"

        for cm, variant in [(hard_cm, "hard (MAP z)"), (soft_cm, "soft (posterior)"),
                            (heur_cm, "hard (heuristic baseline)")]:
            if cm is None:
                continue
            print(f"\n  [{variant}]")
            print(header)
            for k, name in enumerate(STATE_NAMES):
                print(_fmt_row(name, cm[k]))

    # ── C confusion ──────────────────────────────────────────────────────────
    for method_name, cr in results.get("c_results", {}).items():
        print(f"\n{SEP}")
        print(f"C confusion — method: {method_name}")
        print(f"  Aggregation: row-normalised CM averaged over {cr['n_homes']} homes")
        print(f"  ({cr['n_ev']} EV, {cr['n_non_ev']} non-EV)  accuracy={cr['accuracy']:.4f}")
        print(f"  Rows = true C, columns = predicted C")
        header = f"  {'':8}  {'no-EV':>7}  {'EV':>7}"
        for cm, variant in [(cr["hard_cm"], "hard (threshold 0.5)"),
                            (cr["soft_cm"], "soft (P̂ as fraction)")]:
            print(f"\n  [{variant}]")
            print(header)
            for k, name in enumerate(["no-EV", "EV"]):
                print(_fmt_row(name, cm[k]))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def format_confusion(cm: np.ndarray, labels: list[str]) -> str:
    n = cm.shape[0]
    header = "          " + "  ".join(f"{l:>8}" for l in labels) + "   total"
    lines = [header]
    for i in range(n):
        row = cm[i]
        lines.append(
            f"  {labels[i]:>6} | " + "  ".join(f"{int(c):>8}" for c in row)
            + f"   {int(row.sum()):>8}"
        )
    col_totals = cm.sum(axis=0)
    lines.append(
        "  " + " " * 6 + " | " + "  ".join(f"{int(c):>8}" for c in col_totals)
        + f"   {int(cm.sum()):>8}"
    )
    return "\n".join(lines)
