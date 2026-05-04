# Inference

All notation follows `graphical_model.tex`. Global parameters are fixed
(from `fit.md`). At test time, only $x^{(n)}_{d,t}$ is observed; we infer:

- $C^{(n)}$ — via the **baseline heuristic** (`models/first_diff_logistic.py`),
  not Gibbs-sampled (see §1).
- $z^{(n)}_{d,t}$, $\alpha^{(n)}$, $\Theta^{(n)}_k$ — via **Gibbs sampling**
  (§2), only for homes with $\hat C^{(n)} = 1$.

The combined emission for total grid power is:

$$x^{(n)}_{d,t} \mid z^{(n)}_{d,t}{=}k \sim \mathcal{N}\!\left(\Theta^{(n)}_k + \alpha^{(n)} \rho_t,\;\; (\sigma^{\textrm{EV}}_k)^2 + (\sigma^{\textrm{Non-EV}}_t)^2\right)$$

This follows from $x = x^{\textrm{EV}} + x^{\textrm{Non-EV}}$ with independent
noise; the variance is the sum of the two emission variances. Define the
shorthand $\sigma^2_{k,t} \equiv (\sigma^{\textrm{EV}}_k)^2 + (\sigma^{\textrm{Non-EV}}_t)^2$.

---

## 1. EV ownership $\hat C^{(n)}$ — heuristic, not Gibbs

Apply `first_diff_logistic.predict` to each home's total-load sequence to
get a binary $\hat C^{(n)} \in \{0, 1\}$.

**Why not Gibbs over $C^{(n)}$?** The chain is degenerate: if any
$z^{(n)}_{d,t} \neq \mathtt{off}$ then $C^{(n)}{=}1$ deterministically,
but the converse fails — the chain can get stuck at $C{=}0$, $z{\equiv}\mathtt{off}$
even when the home is an EV home. The heuristic short-circuits this.

If $\hat C^{(n)} = 0$: set $z^{(n)}_{d,t} \equiv \mathtt{off}$, skip Gibbs.

---

## 2. Gibbs sampler (EV homes only)

For each home $n$ with $\hat C^{(n)}{=}1$, sample from the joint posterior:

$$p\!\left(\alpha^{(n)},\, \Theta^{(n)}_{\mathtt{low}},\, \Theta^{(n)}_{\mathtt{high}},\, \{z^{(n)}_{d,1:T}\}_{d=1}^{D^{(n)}} \;\middle|\; x^{(n)}_{1:D,1:T}\right)$$

**Initialization.** $\alpha^{(n)} = \mu_\alpha$, $\Theta^{(n)}_k = \mu_{\Theta_k}$,
$z^{(n)}_{d,t} = \mathtt{off}$ for all $(d,t)$.

**Schedule.** $S_{\textrm{burn}} = 200$ burn-in iters, then $S = 500$ retained iters.
At each iteration, run blocks 1 → 2 → 3 in order. After burn-in, accumulate
posterior counts (see §3).

### Block 1 — Sample $z^{(n)}_{d,1:T}$ via FFBS (vectorized over $D$)

For each iteration, precompute (shape $(K, T)$):

$$\sigma^2_{k,t} = (\sigma^{\textrm{EV}}_k)^2 + (\sigma^{\textrm{Non-EV}}_t)^2,\quad
\mu_{k,t} = \Theta^{(n)}_k + \alpha^{(n)} \rho_t$$

Per-day, per-state log-emission (shape $(D, T, K)$):

$$\log p(x^{(n)}_{d,t} \mid z{=}k) = -\tfrac{1}{2}\log(2\pi \sigma^2_{k,t}) - \frac{(x^{(n)}_{d,t} - \mu_{k,t})^2}{2 \sigma^2_{k,t}}$$

**Forward pass** (log-space; $f_t \in \mathbb{R}^{D \times K}$ is the
normalized log-filter at time $t$):

$$f_0 = \log\pi_z + \log p(x_{:,0} \mid z); \quad f_0 \mathrel{-}= \mathrm{LSE}_k(f_0)$$

For $t = 1, \ldots, T-1$:

$$f_t[d, k'] = \log p(x_{d,t} \mid z{=}k') + \mathrm{LSE}_k\!\left(f_{t-1}[d, k] + \log P_z[k, k']\right)$$

then renormalize $f_t$ to sum to 1 over $k'$ (per $d$).

**Backward pass** (sample). Sample $z_{:, T-1}$ from $\mathrm{Cat}(\exp f_{T-1})$
across all $D$ days simultaneously. For $t = T-2, \ldots, 0$:

$$w[d, k] = \exp f_t[d, k] \cdot P_z[k,\, z_{d, t+1}],\qquad z_{d, t} \sim \mathrm{Cat}\!\left(w[d,:]/\sum_k w[d,k]\right)$$

**Cost.** $O(K^2 T D)$ per iter. With $K{=}3$, $T{=}96$, $D{\le}360$:
< 1ms in vectorized NumPy.

### Block 2 — Sample $\Theta^{(n)}_k$ for $k \in \{\mathtt{low}, \mathtt{high}\}$

Conditional on current $z$ and $\alpha$, observations in state $k$ contribute:

$$x^{(n)}_{d,t} - \alpha^{(n)} \rho_t \;\sim\; \mathcal{N}(\Theta^{(n)}_k, \sigma^2_{k,t})$$

(Note $\sigma^2_{k,t}$ depends only on $t$ when $k$ is fixed.)

Posterior is Gaussian (Gaussian prior × Gaussian likelihood). Let
$\mathcal{T}_k = \{(d,t) : z_{d,t}{=}k\}$.

$$\mathrm{prec}_k = \frac{1}{\sigma_{\Theta_k}^2} + \sum_{(d,t)\in \mathcal{T}_k} \frac{1}{\sigma^2_{k,t}}$$

$$m_k = \frac{1}{\mathrm{prec}_k}\!\left(\frac{\mu_{\Theta_k}}{\sigma_{\Theta_k}^2} + \sum_{(d,t)\in \mathcal{T}_k} \frac{x^{(n)}_{d,t} - \alpha^{(n)} \rho_t}{\sigma^2_{k,t}}\right)$$

$$\Theta^{(n)}_k \sim \mathcal{N}(m_k,\, 1/\mathrm{prec}_k)$$

If $|\mathcal{T}_k| = 0$, draw from the prior.

### Block 3 — Sample $\alpha^{(n)}$

Conditional on current $z$ and $\Theta$, every observation contributes
(over all $(d,t)$):

$$x^{(n)}_{d,t} - \Theta^{(n)}_{z_{d,t}} \;\sim\; \mathcal{N}\!\left(\alpha^{(n)} \rho_t,\, \sigma^2_{z_{d,t}, t}\right)$$

Scalar Gaussian regression onto fixed regressor $\rho$:

$$\mathrm{prec}_\alpha = \frac{1}{\sigma_\alpha^2} + \sum_{d,t} \frac{\rho_t^2}{\sigma^2_{z_{d,t}, t}}$$

$$m_\alpha = \frac{1}{\mathrm{prec}_\alpha}\!\left(\frac{\mu_\alpha}{\sigma_\alpha^2} + \sum_{d,t} \frac{\rho_t (x^{(n)}_{d,t} - \Theta^{(n)}_{z_{d,t}})}{\sigma^2_{z_{d,t}, t}}\right)$$

$$\alpha^{(n)} \sim \mathcal{N}(m_\alpha,\, 1/\mathrm{prec}_\alpha)$$

The double sum over $(d,t)$ has $\sim D \cdot T \approx 35{,}000$ terms.
Vectorized via fancy indexing on $\sigma^2_{z, t}$.

---

## 3. Posterior summaries

After burn-in, accumulate posterior counts incrementally (avoid storing
$S \cdot D \cdot T$ samples):

$$z\text{-counts}[d, t, k] \mathrel{+}= \mathbf{1}[z^{(n,s)}_{d,t} = k]\quad \text{after each post-burn iteration}$$

Final marginals:

$$\hat p(z^{(n)}_{d,t} = k) = \frac{z\text{-counts}[d, t, k]}{S},\qquad \hat z^{(n)}_{d,t} = \arg\max_k\,\hat p(z^{(n)}_{d,t}{=}k)$$

Also retain per-iteration $\alpha$ and $\Theta$ samples (cheap; size $S$ and $S \times K$).

---

## 4. Evaluation metrics

- **EV ownership.** $2\times 2$ confusion matrix of $\hat C^{(n)}$ vs ground-truth $C^{(n)}$ across all evaluation homes.
- **Charging state.** $3\times 3$ confusion matrix of $\hat z^{(n)}_{d,t}$ vs ground-truth $z^{(n)}_{d,t}$ across all evaluation $(n, d, t)$ triples (EV homes only).

For comparison, also report the heuristic's own per-timestep state output
(`first_diff_logistic.predict` returns this) against the same ground truth —
this is the "baseline" against which Gibbs is compared.

---

## 5. Computational budget

- Per-iteration cost on one EV home ($D{\approx}360$, $T{=}96$): FFBS $O(K^2 T D) \approx 3{\times}10^5$ ops, conjugate updates $O(KDT) \approx 10^5$ ops.
- $700$ iters × ~9 EV homes ≈ a few minutes of pure NumPy on a laptop.
- Memory per home: $z$-counts is $(D, T, K)$ floats = $\le 105\mathrm{K}$ floats ≈ 1 MB.
