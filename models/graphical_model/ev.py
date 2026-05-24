"""EV submodel — §1 of specs/model.md.

Split into two sections:
    1.  Fit-time     fit_hmm, fit_charging_em
    2.  Sample-time  sample_theta_k, hmm_forward, hmm_backward_sample, ffbs

Latents handled here:
    z^(n)_{d,t} ∈ {off, low, high}     — HMM charging state
    Theta^(n)_k                         — per-home mean charging power in state k

Global params estimated here:
    pi_z, P_z                                — HMM initial / transition
    mu_theta, sigma2_theta, sigma2_ev        — charging-magnitude hyperparams
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, truncnorm

from .params import (
    EM_MAX_ITERS, EM_TOL, K, LAPLACE, SIGMA_EV_OFF, STATE_NAMES,
    T, THETA_BOUNDS, THETA_VAR_FLOOR, ModelParams,
)


# ===========================================================================
# 1. FIT-TIME — HMM and charging magnitudes
# ===========================================================================

def fit_hmm(home_arrays: dict, ev_homes: list[int], *, verbose: bool):
    """Fit (pi_z, P_z) by empirical counts on EV homes, with Laplace smoothing.

    Days are independent chains: transitions are counted strictly within days,
    never across day boundaries (see specs/model.md §1.4).
    """
    pi_counts    = np.zeros(K,        dtype=np.float64)
    trans_counts = np.zeros((K, K),   dtype=np.float64)

    for hid in ev_homes:
        z = home_arrays[hid]["z"]
        starts = z[:, 0]                          # initial-state counts at t=0
        for k in range(K):
            pi_counts[k] += int(np.sum(starts == k))
        prev = z[:, :-1].ravel()                  # within-day transitions only
        nxt  = z[:,  1:].ravel()
        for k in range(K):
            mask = (prev == k)
            if not mask.any():
                continue
            for kp in range(K):
                trans_counts[k, kp] += int(np.sum(nxt[mask] == kp))

    pi_z = pi_counts / pi_counts.sum()
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


def fit_charging_em(home_arrays: dict, ev_homes: list[int], *, verbose: bool):
    """One-way Gaussian random-effects EM (specs/model.md §1.6).

    Off-state (k=0) is fixed: mu_theta=0, sigma2_theta=0, sigma2_ev=SIGMA_EV_OFF^2.

    For each k ∈ {low, high}, EM alternates:
      E-step: posterior moments E[Theta^(n)_k], Var[Theta^(n)_k] given current hyperparams
      M-step: closed-form updates for mu_theta_k, sigma2_theta_k, sigma2_ev_k
    """
    mu_theta     = np.zeros(K, dtype=np.float64)
    sigma2_theta = np.zeros(K, dtype=np.float64)
    sigma2_ev    = np.zeros(K, dtype=np.float64)
    sigma2_ev[0] = SIGMA_EV_OFF ** 2

    for k in (1, 2):
        # Per-home sufficient stats for state k: count, sum, sum-of-squares
        n_per_home, S_y, SS_y = [], [], []
        for hid in ev_homes:
            z = home_arrays[hid]["z"]
            x_ev = home_arrays[hid]["x_ev"]
            mask = (z == k)
            n_per_home.append(int(mask.sum()))
            obs = x_ev[mask]
            S_y.append(float(obs.sum()))
            SS_y.append(float((obs ** 2).sum()))
        n_per_home = np.array(n_per_home, dtype=np.float64)
        S_y        = np.array(S_y,        dtype=np.float64)
        SS_y       = np.array(SS_y,       dtype=np.float64)

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
            mu_theta[k]     = 0.0
            sigma2_theta[k] = 0.0
            sigma2_ev[k]    = 0.01
            continue

        # ANOVA-style initialization
        active = n_per_home > 0
        theta_hat = np.where(active, S_y / np.maximum(n_per_home, 1), 0.0)

        N_k = n_per_home.sum()
        SS_within = SS_y[active].sum() - (n_per_home[active] * theta_hat[active] ** 2).sum()
        denom = N_k - active.sum()
        sigma2_ev_k    = SS_within / max(denom, 1.0)
        var_theta_hat  = float(np.var(theta_hat[active], ddof=1)) if active.sum() > 1 else 0.0
        mean_inv_n     = float(np.mean(1.0 / n_per_home[active]))
        sigma2_theta_k = max(0.0, var_theta_hat - sigma2_ev_k * mean_inv_n)
        mu_theta_k     = float(theta_hat[active].mean())

        if verbose:
            print(f"  ANOVA init: mu={mu_theta_k:.4f}, "
                  f"sigma_Theta={np.sqrt(sigma2_theta_k):.4f}, "
                  f"sigma^EV={np.sqrt(sigma2_ev_k):.4f}")

        prev_loglik = -np.inf
        for it in range(EM_MAX_ITERS):
            sig2_theta = max(sigma2_theta_k, THETA_VAR_FLOOR)

            # E-step: posterior moments of Theta_k for each home
            prec       = 1.0 / sig2_theta + n_per_home / sigma2_ev_k
            E_theta    = (mu_theta_k / sig2_theta + S_y / sigma2_ev_k) / prec
            Var_theta  = 1.0 / prec
            E_theta2   = Var_theta + E_theta ** 2

            loglik = _charging_loglik(n_per_home, theta_hat, S_y, SS_y,
                                       mu_theta_k, sig2_theta, sigma2_ev_k, active)

            # M-step
            mu_theta_k     = float(np.mean(E_theta))
            sigma2_theta_k = float(np.mean(Var_theta + (E_theta - mu_theta_k) ** 2))
            sigma2_ev_k    = float(
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

        mu_theta[k]     = mu_theta_k
        sigma2_theta[k] = max(sigma2_theta_k, 0.0)
        sigma2_ev[k]    = sigma2_ev_k

        if verbose:
            lb, ub = THETA_BOUNDS[k]
            sd = np.sqrt(max(sigma2_theta_k, THETA_VAR_FLOOR))
            # Prior mass inside the state-magnitude bounds [lb, ub] — a sanity check.
            # Low values (<~0.5) mean the untruncated prior puts most of its
            # mass outside the bound; the truncated posterior is still OK but
            # the prior is barely informative inside the band.
            mass_in = float(norm.cdf((ub - mu_theta_k) / sd) - norm.cdf((lb - mu_theta_k) / sd))
            print(f"  Theta_{STATE_NAMES[k]} bound [{lb}, {ub}]: "
                  f"prior mass in band = {mass_in:.3f} "
                  f"(mu={mu_theta_k:.3f}, sigma_Theta={sd:.3f})")

    return mu_theta, sigma2_theta, sigma2_ev


def _charging_loglik(n, theta_hat, S_y, SS_y, mu, sig2_theta, sig2_ev, active):
    """Marginal log-likelihood of one-way Gaussian RE model (sum over homes)."""
    n_a  = n[active]
    th_a = theta_hat[active]
    SS_a = SS_y[active]
    within_ss = SS_a - n_a * th_a ** 2

    ll = -0.5 * np.sum(
        n_a * np.log(2 * np.pi)
        + (n_a - 1) * np.log(sig2_ev)
        + np.log(sig2_ev + n_a * sig2_theta)
        + within_ss / sig2_ev
        + n_a * (th_a - mu) ** 2 / (sig2_ev + n_a * sig2_theta)
    )
    return float(ll)


# ===========================================================================
# 2. SAMPLE-TIME — Gibbs blocks
# ===========================================================================

# --- Block 2: Theta_k  (truncated-Gaussian conjugate, per state k) -----------

def sample_theta_k(
    x:      np.ndarray,    # (D, T)
    z:      np.ndarray,    # (D, T)
    eta:    np.ndarray,    # (T,)
    omega2: np.ndarray,    # (T,)
    params: ModelParams,
    k:      int,
    rng,
) -> float:
    """Sample Theta_k from its conditional truncated-Normal (specs/model.md §1.5).

        x[d,t] - eta[t]  ~  N( Theta_k, sigma2_ev[k] + omega2[t] )   for (d,t) ∈ T_k
        Theta_k          ~  N( mu_theta_k, sigma2_theta_k ) · 1[Theta_k ∈ [lb, ub]]

    The truncation indicator passes through Gaussian conjugacy unchanged, so the
    posterior is the same untruncated-conjugate Normal truncated to [lb, ub].
    Heteroscedastic across (d,t): conditional on z[d,t]=k, variance depends only
    on t, but t varies within the masked set.
    """
    sigma2_ev_k = params.sigma2_ev[k]
    sig2_prior  = max(params.sigma2_theta[k], THETA_VAR_FLOOR)
    lb, ub      = THETA_BOUNDS[k]

    mask = (z == k)
    if not mask.any():                                    # no obs in state k: draw from truncated prior
        return _truncnorm_sample(params.mu_theta[k], np.sqrt(sig2_prior), lb, ub, rng)

    var_t     = sigma2_ev_k + omega2                      # (T,) heteroscedastic per t
    inv_var_t = 1.0 / var_t

    # Sufficient statistics: Σ_{(d,t)∈T_k} 1/var_t and Σ_{(d,t)∈T_k} (x-eta)/var_t
    r = x - eta[None, :]
    S_inv_var = (mask * inv_var_t[None, :]).sum()
    S_r       = (mask * r * inv_var_t[None, :]).sum()

    prec = 1.0 / sig2_prior + S_inv_var
    m    = (params.mu_theta[k] / sig2_prior + S_r) / prec
    return _truncnorm_sample(m, np.sqrt(1.0 / prec), lb, ub, rng)


def _truncnorm_sample(mean: float, sd: float, lb: float, ub: float, rng) -> float:
    """Draw one sample from N(mean, sd^2) truncated to [lb, ub]. lb/ub may be inf."""
    a = (lb - mean) / sd
    b = (ub - mean) / sd
    return float(truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng))


# --- Block 1: z  (FFBS — forward filter, backward sample) -------------------

def hmm_forward(
    x:      np.ndarray,   # (D, T)
    theta:  np.ndarray,   # (K,)
    eta:    np.ndarray,   # (T,)
    omega2: np.ndarray,   # (T,)
    params: ModelParams,
    log_pi: np.ndarray,
    log_P:  np.ndarray,
) -> tuple[np.ndarray, float]:
    """HMM forward pass, vectorized over days.

    Emission per (d, t, k):
        N( x[d,t] ; theta[k] + eta[t], sigma2_ev[k] + omega2[t] )

    Returns
    -------
    log_f  : (D, T, K) normalised log filter messages
    log_Z1 : log p(x | C=1, eta, theta, omega2) accumulated from per-step
             log-normalisation constants — the marginal likelihood that the
             collapsed C-step uses.
    """
    D = x.shape[0]
    combined_var = params.sigma2_ev[:, None] + omega2[None, :]   # (K, T)
    inv_2var     = 0.5 / combined_var
    log_norm     = -0.5 * np.log(2 * np.pi * combined_var)

    mean_kt  = theta[:, None] + eta[None, :]                     # (K, T)
    diff     = x[:, :, None] - mean_kt.T[None, :, :]             # (D, T, K)
    log_emit = log_norm.T[None, :, :] - diff ** 2 * inv_2var.T[None, :, :]  # (D, T, K)

    log_f  = np.empty((D, T, K), dtype=np.float64)
    log_Z1 = 0.0

    # t=0 — combine with initial distribution
    unnorm_0       = log_pi[None, :] + log_emit[:, 0, :]
    lse_0          = logsumexp(unnorm_0, axis=1)
    log_Z1        += lse_0.sum()
    log_f[:, 0, :] = unnorm_0 - lse_0[:, None]

    # t=1..T-1 — recursion in log-space
    for t in range(1, T):
        log_pred       = logsumexp(log_f[:, t-1, :, None] + log_P[None, :, :], axis=1)
        unnorm_t       = log_emit[:, t, :] + log_pred
        lse_t          = logsumexp(unnorm_t, axis=1)
        log_Z1        += lse_t.sum()
        log_f[:, t, :] = unnorm_t - lse_t[:, None]

    return log_f, log_Z1


def hmm_backward_sample(log_f: np.ndarray, params: ModelParams, rng) -> np.ndarray:
    """Backward sampling pass given pre-computed forward messages.

    log_f : (D, T, K) from hmm_forward
    Returns z : (D, T) sampled state sequence.
    """
    D   = log_f.shape[0]
    z   = np.empty((D, T), dtype=np.int64)
    p_T = np.exp(log_f[:, T-1, :])
    p_T /= p_T.sum(axis=1, keepdims=True)
    z[:, T-1] = _sample_categorical_rows(p_T, rng)

    P_z = params.P_z
    for t in range(T - 2, -1, -1):
        col = P_z[:, z[:, t+1]].T                # (D, K) — P(z_t -> z_{t+1}) for each row
        w   = np.exp(log_f[:, t, :]) * col
        w  /= w.sum(axis=1, keepdims=True)
        z[:, t] = _sample_categorical_rows(w, rng)

    return z


def ffbs(
    x:      np.ndarray,   # (D, T)
    theta:  np.ndarray,   # (K,)
    eta:    np.ndarray,   # (T,)
    omega2: np.ndarray,   # (T,)
    params: ModelParams,
    log_pi: np.ndarray,
    log_P:  np.ndarray,
    rng,
) -> tuple[np.ndarray, float]:
    """Forward-filter backward-sample for z. Returns (z, log_Z1)."""
    log_f, log_Z1 = hmm_forward(x, theta, eta, omega2, params, log_pi, log_P)
    return hmm_backward_sample(log_f, params, rng), log_Z1


def _sample_categorical_rows(probs: np.ndarray, rng) -> np.ndarray:
    """Draw one categorical sample per row of a (D, K) probability matrix."""
    cum = np.cumsum(probs, axis=1)
    u   = rng.random(probs.shape[0])[:, None]
    return np.argmax(cum > u, axis=1)
