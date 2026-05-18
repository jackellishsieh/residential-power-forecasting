# Collapsed Gibbs for C: Design Document

## Background

The current sampler (`infer_home`) has four Gibbs blocks per iteration:

1. **z | x, α, Θ** — "mixture FFBS": run the HMM forward–backward under C=1, then
   flip a coin weighted by `p(C=1)·p(x|C=1) vs p(C=0)·p(x|C=0)` to decide whether
   z = z_FFBS or z = all-off.
2. **C | z** — sample C from a logistic regression on transitions-per-day (a heuristic).
3. **Θ_k | z, α, x** — conjugate Gaussian.
4. **α | z, Θ, x** — conjugate Gaussian.

### Problem with Block 2

Block 2 samples C from a *heuristic* proxy: a logistic regression trained on the
number of within-day state transitions. This was introduced because the naive rule
`C = any(z ≠ off)` fires for every home (since `P(z = all-off | D≈360, C=1) ≈ 0`),
making C ≡ 1 always.

The heuristic works empirically but it does **not** correspond to the correct
conditional distribution. The chain is therefore not a valid Gibbs sampler for the
intended model.

---

## Proposed Fix: Collapsed Gibbs for C

### Key insight

`p(C | x, α, Θ)` can be computed *exactly* by marginalizing over z:

```
p(C = 1 | x, α, Θ)  ∝  p(C=1) · p(x | C=1, α, Θ)
p(C = 0 | x, α, Θ)  ∝  p(C=0) · p(x | C=0, α)
```

- `p(x | C=1, α, Θ)` is the **HMM marginal likelihood**, which is exactly the product
  of per-step normalisation constants accumulated during the forward pass (already
  implemented in `_ffbs` as `log_Z1`).
- `p(x | C=0, α)` = `p(x | z ≡ off, α)` is already implemented as
  `_compute_loglik_c0`.

So the correct collapsed Gibbs step for C costs nothing extra: the quantity we need
(`log_Z1`) is a by-product of the forward pass we already run.

### New block ordering

```
1. C | x, α, Θ   — collapsed Bernoulli (forward pass → log_Z1 → exact posterior)
2. z | C, x, α, Θ — FFBS backward sample if C=1; z = all-off if C=0
3. Θ_k | z, α, x  — conjugate Gaussian (unchanged)
4. α | z, Θ, x   — conjugate Gaussian (unchanged)
```

Blocks 3 and 4 are identical to the current implementation.

---

## Implementation Plan

### 1. Split `_ffbs` into two functions

**`_hmm_forward(x, theta, alpha, params, log_pi, log_P) → (log_f, log_Z1)`**

Runs only the forward pass. Returns:
- `log_f`: `(D, T, K)` array of normalised log filter messages
- `log_Z1`: scalar, `log p(x | C=1, α, Θ)` = sum of per-step log-normalisation constants

**`_hmm_backward_sample(log_f, params, rng) → z`**

Runs the backward sampling pass given pre-computed forward messages.
Returns `z`: `(D, T)` int array. Identical logic to the current backward pass in `_ffbs`.

The existing `_ffbs` can be kept as a thin wrapper calling both, or left untouched for
the old `infer_home`.

### 2. New function `infer_home_collapsed`

Signature is identical to `infer_home` **minus**:
- `initial_c` (removed: C is sampled fresh every iteration from its collapsed
  posterior, so the warm-start has no effect)
- `c_logistic_model` (removed: no longer needed)

`initial_z` is also irrelevant in the new scheme (z is overwritten by Block 2 before
it affects anything else), but can be retained for API compatibility.

**Block 1 (new)**:
```python
log_f, log_Z1 = _hmm_forward(home_x, theta, alpha, params, log_pi, log_P)
log_Z0 = _compute_loglik_c0(home_x, alpha, params)
log_w1 = np.log(params.p_C + 1e-300) + log_Z1
log_w0 = np.log(1 - params.p_C + 1e-300) + log_Z0
p_c1   = float(np.exp(log_w1 - np.logaddexp(log_w1, log_w0)))
c      = int(rng.random() < p_c1)
```

**Block 2 (new)**:
```python
if c == 1:
    z = _hmm_backward_sample(log_f, params, rng)
else:
    z = np.zeros((D, T), dtype=np.int64)
```

### 3. Keep old `infer_home` unchanged

The old function is kept as-is for A/B comparison. `infer_all` and other callers are
not touched.

A new `infer_all_collapsed` wrapper calls `infer_home_collapsed` instead.

---

## Expected Benefits

| Property | Old (`infer_home`) | New (`infer_home_collapsed`) |
|---|---|---|
| C sampler validity | Heuristic (logistic on transitions) | Exact collapsed Gibbs |
| Mixing for C | Depends on z ↔ C correlation | Better: C sampled from true marginal |
| Mixing for z | Good (FFBS) | Identical |
| Backward pass when C=0 | Always run (then discarded) | Skipped — efficiency gain for non-EV homes |
| External dependency | sklearn LogisticRegression | None |

---

## Performance Comparison

To compare the two samplers, run both on the same test set and compare:

- **C classification accuracy** (primary metric): does collapsed Gibbs classify EV vs
  non-EV homes more accurately?
- **z state accuracy** on EV homes: does it recover charging states better?
- **Mixing diagnostics**: trace plots and ESS for α, Θ, z state occupancy.
- **Runtime**: collapsed Gibbs may be faster for non-EV homes (skips backward pass).

Suggested experiment: run `infer_all` (old) and `infer_all_collapsed` (new) on the
same held-out set, with the same `S_burn` and `S`, seeded identically, then call
`evaluate` on both result dicts.