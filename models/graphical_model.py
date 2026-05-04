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

    # Per-sample C indicator: 1 if any z[d,t] != off in that sample
    c_from_z_samples: np.ndarray | None = None          # (S,) binary

    # Per-sample within-day z transition rate (for heuristic-based C estimation)
    z_transitions_per_day_samples: np.ndarray | None = None  # (S,) float

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
    C_hat: int,
    *,
    S_burn: int = 200,
    S: int = 500,
    rng: np.random.Generator | None = None,
    home_id: int = -1,
    verbose: bool = True,
    record_traces: bool = True,
) -> HomeInference:
    """Run Gibbs inference for one home.

    home_x       : (D, T) total load.
    C_hat        : if 0, skip Gibbs and return z≡off.  Pass 1 to run Gibbs
                   regardless and derive C probability from the sampled z's.
    record_traces: store full per-iteration alpha/theta/loglik traces for
                   convergence diagnostics (adds ~negligible memory for 700 iters).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    D, T_ = home_x.shape
    assert T_ == T, f"expected T={T}, got {T_}"

    if C_hat == 0:
        if verbose:
            print(f"  [home {home_id}] C_hat=0 → z ≡ off (no Gibbs)")
        return HomeInference(
            home_id=home_id,
            C_hat=0,
            z_hat=np.zeros((D, T), dtype=np.int64),
            S_burn=S_burn,
        )

    if verbose:
        print(f"  [home {home_id}] C_hat=1, D={D} → "
              f"running Gibbs ({S_burn} burn-in + {S} retained)")

    # ── initialise ────────────────────────────────────────────────────────────
    alpha = params.mu_alpha
    theta = params.mu_theta.copy()
    z     = np.zeros((D, T), dtype=np.int64)

    log_pi = np.log(params.pi_z + 1e-300)
    log_P  = np.log(params.P_z  + 1e-300)

    # ── storage ───────────────────────────────────────────────────────────────
    n_total = S_burn + S
    z_counts                       = np.zeros((D, T, K), dtype=np.float64)
    alpha_samples                  = np.zeros(S,         dtype=np.float64)
    theta_samples                  = np.zeros((S, K),    dtype=np.float64)
    c_from_z_samples               = np.zeros(S,         dtype=np.int8)
    z_transitions_per_day_samples  = np.zeros(S,         dtype=np.float64)

    if record_traces:
        alpha_trace     = np.zeros(n_total,       dtype=np.float64)
        theta_trace     = np.zeros((n_total, K),  dtype=np.float64)
        state_occ_trace = np.zeros((n_total, K),  dtype=np.float64)
        loglik_trace    = np.zeros(n_total,       dtype=np.float64)
    else:
        alpha_trace = theta_trace = state_occ_trace = loglik_trace = None

    # ── main loop ─────────────────────────────────────────────────────────────
    t_start = time.time()

    for it in range(n_total):
        # Block 1 — FFBS: sample z
        z = _ffbs(home_x, theta, alpha, params, log_pi, log_P, rng)

        # Block 2 — sample Theta_k (k = low, high); off is fixed at 0
        for k in (1, 2):
            theta[k] = _sample_theta_k(home_x, z, alpha, params, k, rng)

        # Block 3 — sample alpha
        alpha = _sample_alpha(home_x, z, theta, params, rng)

        # ── record traces (every iteration) ───────────────────────────────────
        if record_traces:
            alpha_trace[it]        = alpha
            theta_trace[it]        = theta
            state_occ_trace[it]    = [(z == k).mean() for k in range(K)]
            loglik_trace[it]       = _compute_loglik(home_x, z, theta, alpha, params)

        # ── accumulate post-burn-in quantities ────────────────────────────────
        if it >= S_burn:
            s_idx = it - S_burn
            alpha_samples[s_idx] = alpha
            theta_samples[s_idx] = theta
            for k in range(K):
                z_counts[:, :, k] += (z == k)
            # C probability: 1 if any z != off in this sample
            c_from_z_samples[s_idx] = int(np.any(z != 0))
            # Within-day transition rate for heuristic-based C estimation
            z_transitions_per_day_samples[s_idx] = float(
                (np.diff(z, axis=1) != 0).sum() / D
            )

        # ── verbose progress ──────────────────────────────────────────────────
        if verbose and (it < 3 or it == S_burn or (it + 1) % 100 == 0):
            phase   = "burn-in" if it < S_burn else "keep  "
            elapsed = time.time() - t_start

            # Running-mean convergence signal: change in alpha mean over last 50 retained
            if it >= S_burn + 50:
                recent_alpha = alpha_samples[max(0, s_idx - 49) : s_idx + 1]
                delta_pct = abs(recent_alpha[-1] - recent_alpha[0]) / (recent_alpha.std() + 1e-9) * 100
                conv_flag = f"  Δα/σ={delta_pct:.1f}%"
            else:
                conv_flag = ""

            print(f"    iter {it+1:4d}/{n_total} [{phase}]  "
                  f"α={alpha:.3f}  Θ_low={theta[1]:.3f}  Θ_high={theta[2]:.3f}  "
                  f"logL={loglik_trace[it] if record_traces else float('nan'):.1f}  "
                  f"({elapsed:.1f}s){conv_flag}")

    # ── final summaries ───────────────────────────────────────────────────────
    z_marginals      = z_counts / S
    z_hat            = np.argmax(z_marginals, axis=2)
    c_hat_prob       = float(c_from_z_samples.mean())

    if verbose:
        elapsed = time.time() - t_start
        frac = z_marginals.mean(axis=(0, 1))
        print(f"\n  [home {home_id}] Gibbs done in {elapsed:.1f}s")
        print(f"    posterior state freq : off={frac[0]:.3f}  low={frac[1]:.3f}  high={frac[2]:.3f}")
        print(f"    P̂(C=1) from z samples: {c_hat_prob:.4f}")
        print(f"    α  posterior : mean={alpha_samples.mean():.3f}  std={alpha_samples.std():.4f}")
        for k in (1, 2):
            print(f"    Θ[{STATE_NAMES[k]:>4}] posterior : "
                  f"mean={theta_samples[:, k].mean():.3f}  std={theta_samples[:, k].std():.4f}")

    return HomeInference(
        home_id          = home_id,
        C_hat            = C_hat,
        z_hat            = z_hat,
        z_marginals      = z_marginals,
        alpha_samples    = alpha_samples,
        theta_samples    = theta_samples,
        c_from_z_samples              = c_from_z_samples,
        z_transitions_per_day_samples = z_transitions_per_day_samples,
        alpha_trace      = alpha_trace,
        theta_trace      = theta_trace,
        state_occ_trace  = state_occ_trace,
        loglik_trace     = loglik_trace,
        S_burn           = S_burn,
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
    combined_var = params.sigma2_ev[z] + params.sigma2_nonev[None, :]   # (D, T)
    mean_dt      = theta[z] + alpha * params.rho[None, :]               # (D, T)
    ll           = -0.5 * (np.log(2 * np.pi * combined_var) + (x - mean_dt) ** 2 / combined_var)
    return float(ll.sum())


# ---------------------------------------------------------------------------
# Gibbs blocks
# ---------------------------------------------------------------------------

def _ffbs(x, theta, alpha, params, log_pi, log_P, rng):
    """Forward-filter backward-sample. Vectorized over days.

    x: (D, T). Returns z: (D, T) int.
    """
    D = x.shape[0]
    rho = params.rho                                     # (T,)
    sigma2_nonev = params.sigma2_nonev                   # (T,)
    sigma2_ev = params.sigma2_ev                         # (K,)

    # Combined variance per (k, t): (K, T)
    combined_var = sigma2_ev[:, None] + sigma2_nonev[None, :]
    inv_2var = 0.5 / combined_var
    log_norm = -0.5 * np.log(2 * np.pi * combined_var)   # (K, T)

    # mean[k, t] = theta[k] + alpha * rho[t]
    mean_kt = theta[:, None] + alpha * rho[None, :]      # (K, T)

    # log_emit[d, t, k] = log N(x[d,t]; mean[k,t], combined_var[k,t])
    # Compute as (D, T, K)
    diff = x[:, :, None] - mean_kt.T[None, :, :]         # (D, T, K)
    log_emit = log_norm.T[None, :, :] - (diff ** 2) * inv_2var.T[None, :, :]  # (D, T, K)

    # Forward pass
    log_f = np.empty((D, T, K), dtype=np.float64)
    log_f[:, 0, :] = log_pi[None, :] + log_emit[:, 0, :]
    log_f[:, 0, :] -= logsumexp(log_f[:, 0, :], axis=1, keepdims=True)
    for t in range(1, T):
        # log_pred[d, k'] = LSE_k(log_f[d, t-1, k] + log_P[k, k'])
        # broadcast: log_f[..., :, None] (D,K,1) + log_P[None, :, :] (1,K,K)
        log_pred = logsumexp(
            log_f[:, t-1, :, None] + log_P[None, :, :], axis=1
        )  # (D, K)
        log_f[:, t, :] = log_emit[:, t, :] + log_pred
        log_f[:, t, :] -= logsumexp(log_f[:, t, :], axis=1, keepdims=True)

    # Backward sample
    z = np.empty((D, T), dtype=np.int64)
    # t = T-1
    p_last = np.exp(log_f[:, T-1, :])                                   # (D, K)
    p_last /= p_last.sum(axis=1, keepdims=True)
    z[:, T-1] = _sample_categorical_rows(p_last, rng)

    P_z = params.P_z
    for t in range(T - 2, -1, -1):
        # weights[d, k] = exp(log_f[d, t, k]) * P_z[k, z[d, t+1]]
        col = P_z[:, z[:, t+1]].T                                        # (D, K)
        w = np.exp(log_f[:, t, :]) * col
        w /= w.sum(axis=1, keepdims=True)
        z[:, t] = _sample_categorical_rows(w, rng)

    return z


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
    C_hat_dict: dict[int, int],
    *,
    S_burn: int = 200,
    S: int = 500,
    seed: int = 0,
    verbose: bool = True,
) -> dict[int, HomeInference]:
    """Run infer_home over every home in df. C_hat_dict: {home_id: 0|1}."""
    if verbose:
        print("=" * 60)
        print("INFERENCE: Gibbs over all homes")
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

        C_hat = int(C_hat_dict.get(int(hid), 0))
        if verbose:
            print(f"\n[{i+1}/{len(homes)}] home {hid} (D={D}, "
                  f"C_hat={C_hat}, ground_truth={int(g['has_ev'].iloc[0])})")

        results[int(hid)] = infer_home(
            x, params, C_hat,
            S_burn=S_burn, S=S,
            rng=rng, home_id=int(hid), verbose=verbose,
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
    heuristic_states: dict[int, np.ndarray] | None = None,
) -> dict:
    """Confusion matrices for C and z, with optional heuristic baseline for z."""
    sorted_df = df.sort_values(["home_id", "day", "time_index"])

    # C confusion: 2x2
    C_hat = []
    C_true = []
    for hid, g in sorted_df.groupby("home_id", sort=False):
        if int(hid) not in inferences:
            continue
        C_hat.append(inferences[int(hid)].C_hat)
        C_true.append(int(g["has_ev"].iloc[0]))
    C_hat = np.array(C_hat)
    C_true = np.array(C_true)
    C_cm = _confusion(C_true, C_hat, n_classes=2)

    # z confusion: 3x3 (EV homes only)
    z_hat_all = []
    z_true_all = []
    z_heur_all = []
    z_true_heur_all = []
    for hid, g in sorted_df.groupby("home_id", sort=False):
        if int(g["has_ev"].iloc[0]) != 1:
            continue
        if int(hid) not in inferences:
            continue
        D = len(g) // T
        z_true = g["charge_state"].to_numpy().reshape(D, T)
        inf = inferences[int(hid)]
        z_hat_all.append(inf.z_hat.ravel())
        z_true_all.append(z_true.ravel())
        if heuristic_states is not None and int(hid) in heuristic_states:
            heur = heuristic_states[int(hid)]
            # heuristic gives a flat array; align to ground truth length
            min_len = min(len(heur), z_true.size)
            z_heur_all.append(heur[:min_len])
            z_true_heur_all.append(z_true.ravel()[:min_len])

    z_hat_arr = np.concatenate(z_hat_all) if z_hat_all else np.array([], dtype=int)
    z_true_arr = np.concatenate(z_true_all) if z_true_all else np.array([], dtype=int)
    z_cm = _confusion(z_true_arr, z_hat_arr, n_classes=K)

    out = {
        "C_confusion": C_cm,
        "C_accuracy": float((C_hat == C_true).mean()) if len(C_true) else float("nan"),
        "z_confusion": z_cm,
        "z_accuracy": float((z_hat_arr == z_true_arr).mean()) if len(z_true_arr) else float("nan"),
    }

    if z_heur_all:
        z_heur_arr = np.concatenate(z_heur_all)
        z_true_h = np.concatenate(z_true_heur_all)
        out["z_confusion_baseline"] = _confusion(z_true_h, z_heur_arr, n_classes=K)
        out["z_accuracy_baseline"] = float((z_heur_arr == z_true_h).mean())

    return out


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
