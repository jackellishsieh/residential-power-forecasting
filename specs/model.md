# Residential Power Model — Consolidated Reference

This document consolidates the generative model, training procedure, and inference procedure 
into a single per-variable reference. For each parameter or latent variable we
state: its **distribution** under the model, how it is treated at **fit time**,
how it is treated at **inference time**, **why** we chose that approach (and
what alternatives exist), and **where** it is implemented in code.

All code references point into [`models/graphical_model.py`](../models/graphical_model.py)
unless otherwise noted.

---

## 0. Notation & overall structure

### Indices

- $n \in \{1,\dots,N\}$ — home index.
- $d \in \{1,\dots,D^{(n)}\}$ — day within home $n$. **Per-home day count
  varies** (~180 to ~360).
- $t \in \{1,\dots,T\}$ — 15-min interval within a day ($T=96$).
- $k \in \{\texttt{off}, \texttt{low}, \texttt{high}\}$ — EV charging state ($K=3$).

### Generative story

Each home $n$ has a binary EV-ownership indicator $C^{(n)}$. The total grid
power decomposes additively into an EV contribution and a non-EV (background)
contribution:

$$x^{(n)}_{d,t} = x^{\text{EV},(n)}_{d,t} + x^{\text{Non-EV},(n)}_{d,t}.$$

Each contribution is Gaussian conditional on per-home latents. The marginal
observation model used at inference is:

$$x^{(n)}_{d,t} \mid z^{(n)}_{d,t}{=}k \sim \mathcal{N}\!\left(\Theta^{(n)}_k + \alpha^{(n)}\rho_t,\ (\sigma^{\text{EV}}_k)^2 + (\sigma^{\text{Non-EV}}_t)^2\right),$$

with the shorthand $\sigma^2_{k,t} \equiv (\sigma^{\text{EV}}_k)^2 + (\sigma^{\text{Non-EV}}_t)^2$.

### Phase summary

- **Fit** ([`fit()`](../models/graphical_model.py#L129)) uses fully-labeled
  training data — $C^{(n)}$, $z^{(n)}_{d,t}$, $x^{\text{EV}}$, $x^{\text{Non-EV}}$,
  and $x$ are all observed — to estimate the ~11 global parameter blocks in
  closed form (apart from a short EM loop for the charging-magnitude block).
  Output is a [`ModelParams`](../models/graphical_model.py#L37) dataclass.
- **Inference** ([`infer_all()`](../models/graphical_model.py#L816)) sees only
  $x^{(n)}_{d,t}$. EV ownership $C^{(n)}$ is predicted by a heuristic
  ([`first_diff_logistic.predict`](../models/first_diff_logistic.py#L94)); the
  remaining per-home latents ($z, \alpha, \Theta$) are Gibbs-sampled per home
  via [`infer_home()`](../models/graphical_model.py#L490).
- **Evaluation** ([`evaluate()`](../models/graphical_model.py#L913),
  [`print_evaluation()`](../models/graphical_model.py#L1088)) compares
  $\hat C^{(n)}$ and $\hat z^{(n)}_{d,t}$ to ground truth via confusion matrices.

---

## 1. EV submodel

### 1.1 $C^{(n)}$ — EV ownership indicator

**Distribution.** $C^{(n)} \stackrel{\text{iid}}{\sim} \text{Bernoulli}(p_C)$.

**Fit.** Observed in training. Used only to filter the EV-conditional blocks
(HMM and charging-magnitude estimators run on $\mathcal{N}^+ = \{n : C^{(n)}{=}1\}$).

**Inference.** Predicted by a **heuristic baseline**, not Gibbs-sampled.
[`first_diff_logistic.predict`](../models/first_diff_logistic.py#L94) returns
a binary $\hat C^{(n)}$ from the home's total-load sequence; if
$\hat C^{(n)}{=}0$ we set $z^{(n)}_{d,t}\equiv\texttt{off}$ and skip Gibbs.

**Why heuristic, not Gibbs?** The Markov-blanket chain over $C^{(n)}$ is
degenerate: if any $z^{(n)}_{d,t}\neq\texttt{off}$ then $C^{(n)}=1$ deterministically,
but the converse fails — the sampler can get stuck at $C=0$, $z\equiv\texttt{off}$
even when the home is actually an EV home, since flipping $C$ to 1 with all
$z$'s off has no likelihood reward. The heuristic short-circuits this trap.
An alternative would be a tempered/reversible-jump scheme that proposes joint
$(C, z)$ moves, but the heuristic is much simpler and empirically reliable.

**Code.** [`first_diff_logistic.py`](../models/first_diff_logistic.py) (full
detector — tune via [`tune()`](../models/first_diff_logistic.py#L71), apply via
[`predict()`](../models/first_diff_logistic.py#L94)). Dispatch happens inside
[`infer_home()`](../models/graphical_model.py#L490).

### 1.2 $p_C$ — EV prevalence

**Distribution.** Global scalar in $[0,1]$ (point-estimated).

**Fit.** Empirical mean $\hat p_C = \tfrac{1}{N}\sum_n C^{(n)}$. This is the
multinomial/Bernoulli MLE; no smoothing needed at this scale. Implemented in
[`fit()`](../models/graphical_model.py#L129).

**Inference.** Unused — EV ownership is decided by the heuristic, not by a
posterior weighted by $p_C$. (Retained in `ModelParams` for completeness and
future use, e.g. if we revisit a fully-Bayesian $C^{(n)}$ scheme.)

**Why this choice?** Trivial closed-form MLE; alternatives (Beta-Bernoulli
posterior mean, hierarchical pooling across cohorts) would change estimates
by less than the standard error from $N$ alone.

### 1.3 $z^{(n)}_{d,t}$ — charging state

**Distribution.** Per home/day, the sequence $(z^{(n)}_{d,t})_{t=0}^{T-1}$ is
a **daily-reset Markov chain** with initial distribution $\pi_z$ and
transition matrix $P_z$, both conditioned on $C^{(n)}$. If $C^{(n)}=0$,
$z\equiv\texttt{off}$.

**Fit.** Observed in training; nothing to estimate for $z$ itself (the chain's
*parameters* $\pi_z, P_z$ are estimated from these labels — see §1.4).

**Inference.** Sampled jointly across $t$ for each day via **vectorized FFBS**
(forward filter + backward sample, log-space), as Gibbs block 1 inside
[`infer_home()`](../models/graphical_model.py#L490); the core routine is
[`_ffbs()`](../models/graphical_model.py#L702) with backward draws delegated to
[`_sample_categorical_rows()`](../models/graphical_model.py#L758). Cost is
$O(K^2 T D)$ per iter (\<1 ms in NumPy at $K{=}3, T{=}96, D{\le}360$).

**Posterior summary.** After burn-in, counts $z\text{-counts}[d,t,k]$ are
accumulated incrementally to avoid storing $S \cdot D \cdot T$ samples, then
normalized to per-cell marginals; the hard prediction is the argmax over $k$.

**Why FFBS over single-site Gibbs on $z_{d,t}$?** Adjacent states are strongly
coupled through $P_z$ (the chain is sticky in `off`); single-site Gibbs would
mix badly. FFBS is exact-conditional for HMMs and only marginally more code.

**Code.** [`_ffbs()`](../models/graphical_model.py#L702),
[`_sample_categorical_rows()`](../models/graphical_model.py#L758), called from
the Gibbs loop inside [`infer_home()`](../models/graphical_model.py#L490).

### 1.4 $\pi_z, P_z$ — HMM initial distribution and transitions

**Distribution.** $\pi_z\in\Delta^{K-1}$, $P_z\in[0,1]^{K\times K}$ row-stochastic.
Global parameters.

**Fit.** Empirical frequencies over EV homes only ($\mathcal{N}^+$). Days are
treated as **independent chains** — transitions are counted strictly within
days (never across day boundaries):

$$\pi_z[k] = \frac{\sum_{n\in\mathcal{N}^+}\sum_d \mathbf{1}[z^{(n)}_{d,0}{=}k]}{\sum_{n\in\mathcal{N}^+} D^{(n)}}$$

$$P_z[k,k'] = \frac{\lambda + \sum_{n\in\mathcal{N}^+}\sum_d \sum_{t=1}^{T-1} \mathbf{1}[z^{(n)}_{d,t-1}{=}k,\, z^{(n)}_{d,t}{=}k']}{K\lambda + \sum_{n\in\mathcal{N}^+}\sum_d \sum_{t=1}^{T-1} \mathbf{1}[z^{(n)}_{d,t-1}{=}k]}$$

with Laplace smoothing $\lambda = 10^{-3}$.

**Inference.** Read-only — used inside FFBS as the forward-pass transition
weights and backward-pass sampling kernel.

**Why MLE here?** Multinomial counts → empirical frequencies are the MLE; there
is no random-effects structure across homes for the chain parameters (we
deliberately pool — see "alternatives" below). Smoothing is the only safeguard
against unobserved cells, which matter with only $N_{\text{EV}}{=}9$ EV homes.
Alternatives: Dirichlet-multinomial posterior (cosmetically Bayesian, same
estimates at this scale), or per-home transition matrices with hierarchical
pooling (justifiable but adds substantial complexity for a question — "does
this household charge differently?" — that we don't currently care about).

**Code.** [`_fit_hmm()`](../models/graphical_model.py#L257).

### 1.5 $\Theta^{(n)}_k$ — per-home mean charging power in state $k$

**Distribution.** $\Theta^{(n)}_k \stackrel{\text{iid}}{\sim} \mathcal{N}(\mu_{\Theta_k}, \sigma_{\Theta_k}^2)$
for $k\in\{\texttt{low},\texttt{high}\}$, with the off-state pinned:
$\Theta^{(n)}_{\texttt{off}}=0$, $\sigma_{\Theta_{\texttt{off}}}=0$.

**Fit.** Observed in training (since $z$ and $x^{\text{EV}}$ are both labeled,
per-home means $\hat\theta^{(n)}_k = S_y^{(n)} / n^{(n)}_k$ are sufficient
stats). The *hyperparameters* $\mu_{\Theta_k},\sigma_{\Theta_k}^2,\sigma^{\text{EV}}_k$
are jointly fit by short EM — see §1.6.

**Inference.** Gibbs block 2 inside [`infer_home()`](../models/graphical_model.py#L490).
Conditional on current $z,\alpha$, observations in state $k$ are Gaussian with
known mean offset and known per-time variance, giving a Gaussian-prior ×
Gaussian-likelihood update with closed-form posterior:

$$\Theta^{(n)}_k \sim \mathcal{N}(m_k,\, 1/\text{prec}_k),\quad \text{prec}_k = \tfrac{1}{\sigma_{\Theta_k}^2} + \sum_{(d,t)\in\mathcal{T}_k}\tfrac{1}{\sigma^2_{k,t}}.$$

If $|\mathcal{T}_k|=0$ (no observations assigned to state $k$ in the current
$z$), draw from the prior.

**Code.** [`_sample_theta_k()`](../models/graphical_model.py#L789).

### 1.6 $\mu_{\Theta_k}, \sigma_{\Theta_k}, \sigma^{\text{EV}}_k$ — charging-magnitude hyperparameters

**Distribution.** Global point-estimated parameters. $\sigma^{\text{EV}}_k$ is
the within-state, per-timestep emission std around $\Theta^{(n)}_k$. Off-state
is fixed: $\sigma^{\text{EV}}_{\texttt{off}}=10^{-3}$ (small floor for FFBS
numerical stability).

**Fit.** **EM** on the one-way Gaussian random-effects model
([`_fit_charging_em()`](../models/graphical_model.py#L366)). Initialized from
unbalanced ANOVA, then iterated to convergence ($|\Delta\log L|<10^{-6}$ or
100 iters). The E-step computes posterior moments
$\mathbb{E}[\Theta^{(n)}_k], \mathrm{Var}[\Theta^{(n)}_k]$ given current
hyperparameters; the M-step is closed form. Marginal log-likelihood is
monitored via [`_charging_loglik()`](../models/graphical_model.py#L469) and
should be monotone non-decreasing.

**Inference.** Read-only — fixed prior parameters in the conditional
distributions for $\Theta^{(n)}_k$ (§1.5).

**Why EM, not closed-form ANOVA?** Per-home observation counts $n^{(n)}_k$
vary enormously across EV homes — some charge daily, others rarely. Under
unbalanced groups, ANOVA $\neq$ MLE/REML, and gives noticeably biased variance
estimates that would mis-weight homes in inference. EM gives the MLE with
optimal weighting and adds only ~10 lines per iteration; convergence is fast
($\lesssim 50$ iters). Alternatives: REML (similar quality, more code),
fully-Bayesian sampling of hyperparameters (overkill given $N_{\text{EV}}{=}9$
is dominated by data, not prior).

**Code.** [`_fit_charging_em()`](../models/graphical_model.py#L366),
[`_charging_loglik()`](../models/graphical_model.py#L469).

---

## 2. Non-EV (background) submodel

### 2.1 $\alpha^{(n)}$ — per-home background scale

**Distribution.** $\alpha^{(n)} \stackrel{\text{iid}}{\sim} \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)$.

**Fit.** Plug-in OLS projection of the home's day-mean profile onto $\rho$:

$$\hat\alpha^{(n)} = \sum_t \hat\beta^{(n)}_t\,\rho_t, \quad \hat\beta^{(n)}_t = \tfrac{1}{D^{(n)}}\sum_d x^{\text{Non-EV},(n)}_{d,t}.$$

With $D^{(n)}\cdot T \approx$ 17K–35K observations per home, the plug-in is
essentially the MLE — formal EM would shift it by a fraction of a percent.

**Inference.** Gibbs block 3 inside [`infer_home()`](../models/graphical_model.py#L490).
Conditional on current $z,\Theta$, $\alpha^{(n)}$ is the slope of a Gaussian
regression onto the fixed regressor $\rho_t$ with heteroscedastic noise
$\sigma^2_{z_{d,t},t}$:

$$\alpha^{(n)} \sim \mathcal{N}(m_\alpha,\,1/\text{prec}_\alpha),\quad \text{prec}_\alpha = \tfrac{1}{\sigma_\alpha^2} + \sum_{d,t}\tfrac{\rho_t^2}{\sigma^2_{z_{d,t},t}}.$$

The $D\cdot T \approx 35{,}000$-term sum is vectorized via fancy indexing on
$\sigma^2_{z,t}$.

**Code.** Fit: [`_fit_background()`](../models/graphical_model.py#L300).
Inference: [`_sample_alpha()`](../models/graphical_model.py#L789).

### 2.2 $\mu_\alpha, \sigma_\alpha^2$ — cross-home scale hyperparameters

**Distribution.** Global point-estimated parameters.

**Fit.**
$\mu_\alpha = \tfrac{1}{N}\sum_n \hat\alpha^{(n)}$ (empirical mean).
$\sigma_\alpha^2$ uses a **bias correction** that subtracts the noise variance
of the plug-in $\hat\alpha^{(n)}$ — that is, the noise in $\hat\beta^{(n)}$
projected onto $\rho$, scaled by $1/D^{(n)}$:

$$\sigma_\alpha^2 = \max\!\left(0,\ \mathrm{Var}_n(\hat\alpha^{(n)}) - \tfrac{1}{N}\sum_n \tfrac{1}{D^{(n)}}\sum_t \rho_t^2(\sigma^{\text{Non-EV}}_t)^2\right).$$

The correction is small ($O(1/D^{(n)})$) but applied for symmetry with
Step 4 of the EV block and to handle homes with smaller $D^{(n)}$.

**Inference.** Read-only — fixed prior parameters for the $\alpha^{(n)}$
conditional (§2.1).

**Why plug-in + bias correction, not EM?** Per-home data is abundant; the
EM/closed-form difference is in the third decimal. Alternatives we considered:
hierarchically-weighted SVD (negligible difference, more code), full EM
(same answer up to numerical noise, requires iteration).

**Code.** [`_fit_background()`](../models/graphical_model.py#L300).

### 2.3 $\rho_t$ — global normalized intraday background profile

**Distribution.** Global vector $\rho\in\mathbb{R}^T$ with $\|\rho\|_2=1$.

**Fit.** Top right singular vector of the stacked day-mean matrix
$B\in\mathbb{R}^{N\times T}$ (rows are $\hat\beta^{(n)}$). Economy SVD
$B = U\Sigma V^\top$, take $\rho = V_{:,1}$ (unit norm by construction).
Sign is fixed so that $\mathrm{median}_n(\hat\alpha^{(n)}) > 0$, since
background load is non-negative.

**Inference.** Read-only — appears as the regressor in both the observation
mean $\mu_{k,t} = \Theta^{(n)}_k + \alpha^{(n)}\rho_t$ and the $\alpha$
posterior (§2.1).

**Why SVD, not heteroscedastic EM?** With $D^{(n)}\approx$ 200–360 averaging
out per-timestep noise in $\hat\beta$, the heteroscedasticity in $B$'s columns
is at the level of $(\sigma^{\text{Non-EV}}_t)^2 / D^{(n)}$ — small relative to
the $\alpha^{(n)}\rho_t$ signal. Plain SVD vs heteroscedastically-weighted SVD
differ in the third decimal; EM converges to the same answer and requires
iteration. The sign-fix step is the only non-trivial detail.

**Code.** [`_fit_background()`](../models/graphical_model.py#L300).

### 2.4 $\sigma^{\text{Non-EV}}_t$ — per-time background emission noise

**Distribution.** Global vector in $\mathbb{R}^T_{>0}$.

**Fit.** Pooled residual variance against the per-home fitted mean, computed
from **individual observations** (not day-means):

$$(\sigma^{\text{Non-EV}}_t)^2 = \tfrac{1}{\sum_n D^{(n)}}\sum_n\sum_d \left(x^{\text{Non-EV},(n)}_{d,t} - \hat\alpha^{(n)}\rho_t\right)^2.$$

**Why individual obs, not residuals of $\hat\beta$?** Residuals of day-means
would estimate $\mathrm{Var}(\hat\beta^{(n)}_t - \alpha^{(n)}\rho_t) =
(\sigma^{\text{Non-EV}}_t)^2 / D^{(n)}$ — off by a factor of $D\approx 250$
(or $\sqrt D\approx 16$ in std). The FFBS emission would be wildly
overconfident, and Gibbs would collapse onto the prior. The individual-obs
estimator uses all $\sum_n D^{(n)}\cdot T \approx$ 1.5M data points.

**Inference.** Read-only — enters $\sigma^2_{k,t}$ as the time-dependent
component of the emission variance.

**Code.** [`_fit_background()`](../models/graphical_model.py#L300).

---

## 3. Total observation

### 3.1 $x^{(n)}_{d,t}$ — total grid power

**Definition.** $x^{(n)}_{d,t} = x^{\text{EV},(n)}_{d,t} + x^{\text{Non-EV},(n)}_{d,t}$
with independent Gaussian summands, so the marginal emission used at inference
is

$$x^{(n)}_{d,t} \mid z^{(n)}_{d,t}{=}k \sim \mathcal{N}\!\left(\Theta^{(n)}_k + \alpha^{(n)}\rho_t,\ \sigma^2_{k,t}\right),$$

with $\sigma^2_{k,t} = (\sigma^{\text{EV}}_k)^2 + (\sigma^{\text{Non-EV}}_t)^2$.

**Fit.** Fully observed in training; both components $x^{\text{EV}}$,
$x^{\text{Non-EV}}$ are also separately observed (training is on labeled data).

**Inference.** The *only* observed variable. Drives the FFBS emission
likelihoods in block 1 and the data terms in blocks 2 and 3.

**Code.** Total-power arrays are assembled per home by
[`_build_home_arrays()`](../models/graphical_model.py#L230); used throughout
[`infer_home()`](../models/graphical_model.py#L490).

---

## 4. Inference loop (cross-cutting)

Per-home Gibbs sampler ([`infer_home()`](../models/graphical_model.py#L490)),
applied only to homes with $\hat C^{(n)}=1$:

- **Initialization.** $\alpha^{(n)} = \mu_\alpha$, $\Theta^{(n)}_k = \mu_{\Theta_k}$,
  $z\equiv\texttt{off}$.
- **Schedule.** $S_{\text{burn}}=200$ burn-in + $S=500$ retained iterations,
  blocks executed in order 1 → 2 → 3 per iteration.
- **Accumulation.** $z$-counts accumulated incrementally post-burn; per-iter
  $\alpha$ and $\Theta$ samples retained (cheap).
- **Log-likelihood tracking.** [`_compute_loglik()`](../models/graphical_model.py#L667)
  for EV homes; [`_compute_loglik_c0()`](../models/graphical_model.py#L681) for
  non-EV homes (closed-form, no Gibbs).

The driver [`infer_all()`](../models/graphical_model.py#L816) iterates over
homes; [`c_prob_from_z_via_heuristic()`](../models/graphical_model.py#L877) and
[`build_heuristic_homes()`](../models/graphical_model.py#L896) bridge the
heuristic detector into the per-home inference pipeline.

**Computational budget.** $\sim 700$ iters $\times \sim 9$ EV homes $\approx$
a few minutes of pure NumPy on a laptop. Memory per home is dominated by
$z$-counts of shape $(D, T, K)$ — under 1 MB.

---

## 5. Evaluation

Implemented in [`evaluate()`](../models/graphical_model.py#L913) and reported
by [`print_evaluation()`](../models/graphical_model.py#L1088). Two confusion
matrices:

- **EV ownership.** $2\times 2$ confusion of $\hat C^{(n)}$ vs true $C^{(n)}$
  across all evaluation homes
  ([`_c_confusion_from_probs()`](../models/graphical_model.py#L1044)).
- **Charging state.** $3\times 3$ confusion of $\hat z^{(n)}_{d,t}$ vs true
  $z^{(n)}_{d,t}$ across all $(n,d,t)$ triples on EV homes (hard:
  [`_per_home_z_confusion_hard()`](../models/graphical_model.py#L1008); soft:
  [`_per_home_z_confusion_soft()`](../models/graphical_model.py#L1023);
  averaging: [`_nanmean_cms()`](../models/graphical_model.py#L1037)).

For comparison, the heuristic's own per-timestep state output
([`first_diff_logistic.predict`](../models/first_diff_logistic.py#L94)) is
also evaluated against the same ground truth — this is the baseline against
which the Gibbs sampler is compared.

---

## 6. Parameter summary table

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $C^{(n)}$ | per-home latent | observed | heuristic | [`first_diff_logistic.predict`](../models/first_diff_logistic.py#L94) |
| $p_C$ | global scalar | empirical mean | unused | [`fit()`](../models/graphical_model.py#L129) |
| $z^{(n)}_{d,t}$ | per-home latent | observed | FFBS (block 1) | [`_ffbs()`](../models/graphical_model.py#L702) |
| $\pi_z, P_z$ | global | smoothed counts | read-only | [`_fit_hmm()`](../models/graphical_model.py#L257) |
| $\Theta^{(n)}_k$ | per-home latent | observed | Gibbs (block 2) | [`_sample_theta_k()`](../models/graphical_model.py#L789) |
| $\mu_{\Theta_k}, \sigma_{\Theta_k}, \sigma^{\text{EV}}_k$ | global | EM | read-only | [`_fit_charging_em()`](../models/graphical_model.py#L366) |
| $\alpha^{(n)}$ | per-home latent | plug-in OLS | Gibbs (block 3) | [`_sample_alpha()`](../models/graphical_model.py#L789) |
| $\mu_\alpha, \sigma_\alpha^2$ | global | mean / bias-corrected var | read-only | [`_fit_background()`](../models/graphical_model.py#L300) |
| $\rho_t$ | global | top right SVD vector | read-only | [`_fit_background()`](../models/graphical_model.py#L300) |
| $\sigma^{\text{Non-EV}}_t$ | global | individual-obs MSE | read-only | [`_fit_background()`](../models/graphical_model.py#L300) |
| $x^{(n)}_{d,t}$ | observed | observed | observed | [`_build_home_arrays()`](../models/graphical_model.py#L230) |
