# Residential Power Model — Consolidated Reference

This document consolidates the generative model, training procedure, and inference procedure 
into a single per-variable reference. For each parameter or latent variable we
state: its **distribution** under the model, how it is treated at **fit time**,
how it is treated at **inference time**, **why** we chose that approach (and
what alternatives exist), and **where** it is implemented in code.

Code now lives in the [`models/graphical_model/`](../models/graphical_model/)
**package** (one module per section: `params.py`, `_data.py`, `ev.py`,
`non_ev_lds.py`, `fit.py`, `inference.py`, `evaluation.py`); references below
point into those modules. (The old monolithic `graphical_model.py` is gone.)

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

Each contribution is Gaussian conditional on per-home latents. Under the
**current** Non-EV submodel — a per-home **daily linear dynamical system**
(LDS), §2 — the marginal observation model used at inference is:

$$x^{(n)}_{d,t} \mid z^{(n)}_{d,t}{=}k,\ z^{\text{LDS},(n)}_d \sim \mathcal{N}\!\left(\Theta^{(n)}_k + (C\,z^{\text{LDS},(n)}_d)_t,\ (\sigma^{\text{EV}}_k)^2 + R_{tt}\right),$$

with the shorthand $\sigma^2_{k,t} \equiv (\sigma^{\text{EV}}_k)^2 + R_{tt}$,
where $z^{\text{LDS},(n)}_d \in \mathbb{R}^L$ is the home's per-day Non-EV latent
and $(C, R)$ are global LDS emission parameters. Two earlier Non-EV submodels
are kept as **deprecated** contrast: the hierarchical-profile PPCA form (§2.4,
replacing $(C z^{\text{LDS}}_d)_t \to \eta^{(n)}_t$ and $R_{tt} \to (\omega^{(n)}_t)^2$)
and the rank-1 scale-shape form (§2.5, $\to \alpha^{(n)}\rho_t$ and
$(\sigma^{\text{Non-EV}}_t)^2$).

> **Implementation status.** The code in
> [`models/graphical_model/`](../models/graphical_model/) implements the LDS
> Non-EV submodel (§2) and the **two-track** inference of §4 (C=0 and C=1
> inferred separately, then compared by model evidence, §5). The PPCA and
> rank-1 Non-EV submodels, and the old collapsed-$C$ Gibbs sampler, are removed
> from code and kept here only as deprecated contrast (§2.4–§2.5, §4.4).

### Phase summary

- **Fit** ([`fit()`](../models/graphical_model/fit.py)) uses fully-labeled
  training data — $C^{(n)}$, $z^{(n)}_{d,t}$, $x^{\text{EV}}$, $x^{\text{Non-EV}}$,
  and $x$ are all observed — and runs four steps: (1) EV prevalence $p_C$
  (closed form), (2) HMM parameters $\pi_z, P_z$ (smoothed counts), (3) the
  Non-EV LDS parameters (EM), (4) charging-magnitude hyperparameters (EM).
  Output is a [`ModelParams`](../models/graphical_model/params.py) dataclass.
- **Inference is two-track** (§4): the two EV hypotheses share no latents or
  parameters, so each home is inferred under both *separately* and the two are
  compared by their model evidence (§5).
  - [`infer_home_c0()`](../models/graphical_model/inference.py) — **C=0**, no
    sampling: the model collapses to the Non-EV LDS and one exact Kalman
    smoother gives the $z^{\text{LDS}}$ posterior plus $\log p(x\mid C{=}0)$.
  - [`infer_home_c1()`](../models/graphical_model/inference.py) — **C=1**,
    three-block Gibbs over $(\Theta, z^{\text{EV}}, z^{\text{LDS}})$.
  - [`infer_home()`](../models/graphical_model/inference.py) /
    [`infer_all()`](../models/graphical_model/inference.py) run both tracks and
    pick $\hat C$ from the evidence comparison.

  All consume only $x^{(n)}_{d,t}$ at inference time.
- **Evaluation** ([`evaluate()`](../models/graphical_model/evaluation.py),
  [`print_evaluation()`](../models/graphical_model/evaluation.py)) compares
  $\hat C^{(n)}$ and $\hat z^{(n)}_{d,t}$ to ground truth via confusion matrices;
  the presentation plots used by the inference notebook live in
  [`notebooks/utils/infer_plots.py`](../notebooks/utils/infer_plots.py).

---

## 1. EV submodel

### 1.1 $C^{(n)}$ — EV ownership indicator

**Distribution.** $C^{(n)} \stackrel{\text{iid}}{\sim} \text{Bernoulli}(p_C)$.

**Fit.** Observed in training. Used only to filter the EV-conditional blocks
(HMM and charging-magnitude estimators run on $\mathcal{N}^+ = \{n : C^{(n)}{=}1\}$).

**Inference.** Decided by a **model-evidence comparison** (§5.1), not sampled.
For each home we compute the two joint evidences
$\log p(x, C{=}0)$ and $\log p(x, C{=}1)$ and set
$\hat C^{(n)} = \mathbb{1}[\log p(x, C{=}1) > \log p(x, C{=}0)]$; the soft
$\hat P(C{=}1 \mid x)$ is their softmax. The first-diff logistic detector is
kept **only as a comparison baseline** in the inference notebook, not as the
model's $C$-predictor.

**Why evidence comparison, not a sampled $C$?** The two hypotheses share no
latents or parameters ($C{=}0$ has only $z^{\text{LDS}}$; $C{=}1$ adds
$z^{\text{EV}}, \Theta$), so there is nothing to gain from alternating between
them in one chain. The old collapsed-$C$ Gibbs (§4.4) did exactly that and
mixed poorly — once the chain committed to a mode it rarely flipped back,
because moving to $C{=}1$ with all $z^{\text{EV}}{=}\texttt{off}$ earns no
likelihood reward. Inferring each track separately and comparing their
evidence sidesteps the trap entirely and is the textbook Bayesian
model-selection quantity.

**Code.** Evidence comparison in [`infer_home()`](../models/graphical_model/inference.py)
(§5.1). The baseline detector is
[`first_diff_logistic.py`](../models/first_diff_logistic.py), bridged in via
[`build_heuristic_homes()`](../models/graphical_model/inference.py).

### 1.2 $p_C$ — EV prevalence

**Distribution.** Global scalar in $[0,1]$ (point-estimated).

**Fit.** Empirical mean $\hat p_C = \tfrac{1}{N}\sum_n C^{(n)}$. This is the
multinomial/Bernoulli MLE; no smoothing needed at this scale. Implemented in
[`fit()`](../models/graphical_model/fit.py).

**Inference.** Now **used**: it is the $C$ prior in the evidence comparison
(§5.1), $\log p(x, C{=}c) = \log p(C{=}c) + \log p(x \mid C{=}c)$, with
$p(C{=}1)=p_C$ and $p(C{=}0)=1-p_C$.

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

**Inference.** Only present under **C=1** (under C=0, $z\equiv\texttt{off}$ and
nothing is sampled). It is **Block A** of the C=1 Gibbs: sampled jointly across
$t$ for each day via **vectorized FFBS** (forward filter + backward sample,
log-space) on the combined-emission likelihood
$\mathcal{N}(\Theta_k + (C z^{\text{LDS}}_d)_t,\ (\sigma^{\text{EV}}_k)^2 + R_{tt})$,
which factorizes across $t$ given the current $z^{\text{LDS}}$. Core routines
[`ev.hmm_forward()`](../models/graphical_model/ev.py),
[`ev.hmm_backward_sample()`](../models/graphical_model/ev.py) and the wrapper
[`ev.ffbs()`](../models/graphical_model/ev.py), called from
[`infer_home_c1()`](../models/graphical_model/inference.py). Cost is
$O(K^2 T D)$ per iter (\<1 ms in NumPy at $K{=}3, T{=}96, D{\le}360$).

**Posterior summary.** After burn-in, counts `z_counts[d,t,k]` are accumulated
incrementally to avoid storing $S \cdot D \cdot T$ samples, then normalized to
per-cell marginals `z_marginals`; the hard prediction `z_hat` is the argmax
over $k$.

**Why FFBS over single-site Gibbs on $z_{d,t}$?** Adjacent states are strongly
coupled through $P_z$ (the chain is sticky in `off`); single-site Gibbs would
mix badly. FFBS is exact-conditional for HMMs and only marginally more code.

**Code.** [`ev.ffbs()`](../models/graphical_model/ev.py), called from the Gibbs
loop inside [`infer_home_c1()`](../models/graphical_model/inference.py).

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
against unobserved cells, which matter with so few EV homes.
Alternatives: Dirichlet-multinomial posterior (cosmetically Bayesian, same
estimates at this scale), or per-home transition matrices with hierarchical
pooling (justifiable but adds substantial complexity for a question — "does
this household charge differently?" — that we don't currently care about).

**Code.** [`ev.fit_hmm()`](../models/graphical_model/ev.py).

### 1.5 $\Theta^{(n)}_k$ — per-home mean charging power in state $k$

**Distribution.** $\Theta^{(n)}_k \stackrel{\text{iid}}{\sim} \mathcal{N}(\mu_{\Theta_k}, \sigma_{\Theta_k}^2) \cdot \mathbf{1}[\Theta^{(n)}_k \in B_k]$
for $k\in\{\texttt{low},\texttt{high}\}$, i.e. a **truncated normal** with
state-specific magnitude bounds

$$B_{\texttt{low}} = [0.1,\ 2]\ \text{kW},\qquad B_{\texttt{high}} = [2,\ \infty)\ \text{kW}.$$

The off-state is pinned: $\Theta^{(n)}_{\texttt{off}}=0$,
$\sigma_{\Theta_{\texttt{off}}}=0$. The bounds encode the **definitional**
semantics of "low" vs. "high" charging (the same cutoffs used to label states
in the training data) directly into the prior — without them, an unconstrained
$\mathcal{N}$ prior makes the two states statistically indistinguishable apart
from the fitted hyperparameters, which is fragile at inference when the data
is ambiguous. The bounds live in `THETA_BOUNDS` in
[`params.py`](../models/graphical_model/params.py).

**Fit.** Observed in training (since $z$ and $x^{\text{EV}}$ are both labeled,
per-home means $\hat\theta^{(n)}_k = S_y^{(n)} / n^{(n)}_k$ are sufficient
stats). The *hyperparameters* $\mu_{\Theta_k},\sigma_{\Theta_k}^2,\sigma^{\text{EV}}_k$
are jointly fit by short EM — see §1.6. The truncation is not modeled in EM
(we treat $(\mu_{\Theta_k},\sigma_{\Theta_k}^2)$ as the parameters of the
*underlying* untruncated Normal). Since labeled $\hat\theta^{(n)}_k$ values
lie inside $B_k$ by construction of the labels, the bias from omitting the
truncation normalizer in the M-step is small; verbose mode prints the
fraction of *prior* probability mass inside $B_k$ as a sanity check.

**Inference.** **Block B** of the C=1 Gibbs in
[`infer_home_c1()`](../models/graphical_model/inference.py).
Conditional on the current $z^{\text{EV}}$ and $z^{\text{LDS}}$, the residuals
$x^{(n)}_{d,t} - (C z^{\text{LDS}}_d)_t$ in state $k$ are Gaussian with known
mean offset $\Theta^{(n)}_k$ and known per-cell variance
$\sigma^2_{k,t} = (\sigma^{\text{EV}}_k)^2 + R_{tt}$, giving a
truncated-Gaussian-prior × Gaussian-likelihood update. The indicator
$\mathbf{1}[\Theta^{(n)}_k \in B_k]$ passes through the Gaussian conjugacy
unchanged: the posterior is the *same* untruncated-conjugate Normal, truncated
to $B_k$:

$$\Theta^{(n)}_k \sim \mathcal{N}(m_k,\, 1/\text{prec}_k) \cdot \mathbf{1}[\Theta^{(n)}_k \in B_k],\quad \text{prec}_k = \tfrac{1}{\sigma_{\Theta_k}^2} + \sum_{(d,t)\in\mathcal{T}_k}\tfrac{1}{\sigma^2_{k,t}}.$$

Sampled via `scipy.stats.truncnorm.rvs` ($O(1)$ per draw, inverse-CDF based).
If $|\mathcal{T}_k|=0$ (no observations assigned to state $k$ in the current
$z$), draw from the truncated prior.

**Why truncate the prior, not the emission?** Truncating the emission
$x^{\text{EV}}_t | z_t{=}k$ would break FFBS marginalization: the convolution
of a truncated $x^{\text{EV}}$ with Gaussian $x^{\text{Non-EV}}$ has no
closed form, so the per-cell emission likelihood used by the HMM forward
pass would need quadrature or moment-matching. Truncating only the per-home
*mean* $\Theta^{(n)}_k$ leaves the conditional structure of every Gibbs block
intact and addresses the most likely failure mode (the per-home mean drifting
out of its semantic band under ambiguous data at inference). Individual
emissions can still fall outside $B_k$ via $\sigma^{\text{EV}}_k$, which is
appropriate — real chargers ramp up and down, and per-instance readings
genuinely can be just below 2 kW even in the "high" state.

**Code.** [`ev.sample_theta_k()`](../models/graphical_model/ev.py) (and
[`ev.theta_k_posterior_params()`](../models/graphical_model/ev.py), which
returns the conditional $(m_k, \mathrm{sd}_k, \text{bounds})$ without drawing —
reused by the Chib estimator, §5.1),
[`ev._truncnorm_sample()`](../models/graphical_model/ev.py).

### 1.6 $\mu_{\Theta_k}, \sigma_{\Theta_k}, \sigma^{\text{EV}}_k$ — charging-magnitude hyperparameters

**Distribution.** Global point-estimated parameters. $\sigma^{\text{EV}}_k$ is
the within-state, per-timestep emission std around $\Theta^{(n)}_k$. Off-state
is fixed: $\sigma^{\text{EV}}_{\texttt{off}}=10^{-3}$ (small floor for FFBS
numerical stability).

**Fit.** **EM** on the one-way Gaussian random-effects model
([`ev.fit_charging_em()`](../models/graphical_model/ev.py)). Initialized from
unbalanced ANOVA, then iterated to convergence ($|\Delta\log L|<10^{-6}$ or
100 iters). The E-step computes posterior moments
$\mathbb{E}[\Theta^{(n)}_k], \mathrm{Var}[\Theta^{(n)}_k]$ given current
hyperparameters; the M-step is closed form. Marginal log-likelihood is
monitored via [`ev._charging_loglik()`](../models/graphical_model/ev.py) and
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

**Code.** [`ev.fit_charging_em()`](../models/graphical_model/ev.py),
[`ev._charging_loglik()`](../models/graphical_model/ev.py).

---

## 2. Non-EV (background) submodel — per-home daily LDS

> §2.1–§2.3 describe the **implemented** model. §2.4 (PPCA hierarchical
> profile) and §2.5 (rank-1 scale-shape) are **deprecated** predecessors,
> removed from code and kept only as contrast.

The Non-EV background of each home is modelled as a **linear dynamical system
(LDS) whose time axis is _days_**. The per-day state and per-day observation are
both $T$-vectors (one entry per 15-min cell), so the model learns how a home's
*daily load shape* drifts from one day to the next:

$$z^{\text{LDS},(n)}_1 \sim \mathcal{N}(\mu_0,\ \Sigma_0),\qquad
z^{\text{LDS},(n)}_d \mid z^{\text{LDS},(n)}_{d-1} \sim \mathcal{N}(A\,z^{\text{LDS},(n)}_{d-1},\ Q),\qquad
x^{\text{Non-EV},(n)}_d \mid z^{\text{LDS},(n)}_d \sim \mathcal{N}(C\,z^{\text{LDS},(n)}_d,\ R).$$

The LDS parameters $(A, C, Q, R, \mu_0, \Sigma_0)$ are **global** (shared across
homes); only the latent sequence $z^{\text{LDS},(n)}_{1:D}$ is per-home. The
**current instantiation** (set at construction, relaxable via EM flags) is

$$A = I_L \ \text{(random walk)},\qquad C = I_T\ (L = T),\qquad Q,\, R,\, \Sigma_0\ \text{diagonal}.$$

With $A=I$ the latent is a **random walk over days** — day $d$'s mean shape is
day $d{-}1$'s plus a Gaussian innovation of covariance $Q$ — and with $C=I$ the
latent *is* the per-day mean profile in observation space. **A diagonal $R$ is
the load-bearing constraint**: it makes the emission factorize across $t$, which
is exactly what lets the EV-side HMM forward pass (§1.3) treat the cells of a day
independently.

### 2.1 $z^{\text{LDS},(n)}_{d}$ — per-home Non-EV latent sequence

**Distribution.** The Gaussian chain above: a $T$-dim random walk across the
home's days, $z^{\text{LDS},(n)}_{1:D}\in\mathbb{R}^{D\times L}$.

**Fit.** Not inferred at fit time — training observes $x^{\text{Non-EV}}$
directly, and only the *parameters* $(A,C,Q,R,\mu_0,\Sigma_0)$ are estimated
(by EM, §2.2). The latent appears only at inference, when $x^{\text{Non-EV}}$ is
hidden inside the total $x$.

**Inference.** Depends on the track:

- **C = 0** — $z^{\text{LDS}}$ is the *only* latent, and its posterior is
  **exact and Gaussian**: one RTS Kalman smoother returns
  $\mathbb{E}[z^{\text{LDS}}_d \mid x]$ and its covariance. No sampling.
  ([`infer_home_c0`](../models/graphical_model/inference.py).)
- **C = 1** — **Block C** of the Gibbs. Conditional on $(z^{\text{EV}}, \Theta)$,
  the residual $x_{d,t} - \Theta_{z^{\text{EV}}_{d,t}}$ is a linear-Gaussian
  observation of $z^{\text{LDS}}_d$ with **per-cell extra noise**
  $(\sigma^{\text{EV}}_{z_{d,t}})^2$ added to $\mathrm{diag}(R)$; one Kalman FFBS
  pass draws the whole day-sequence jointly.
  ([`non_ev_lds.sample_z_lds`](../models/graphical_model/non_ev_lds.py).)

The same Kalman machinery serves three roles, all keyed off the per-cell
`extra_obs_cov` argument: the **exact $C{=}0$ marginal likelihood**
($z^{\text{EV}}\equiv\texttt{off}$, so $\sigma^{\text{EV}}_{\texttt{off}}{}^2$ is
added everywhere), the **Block-C sample** under $C{=}1$, and the
**$z^{\text{LDS}}$-marginal likelihood** used in the $C{=}1$ evidence (§5.1).

**Why an LDS (vs. the deprecated static-profile priors)?** PPCA/rank-1 give each
home a *single* mean profile shared by all its days. Real homes drift — weekday
vs. weekend, seasonal HVAC, occupancy changes — and a static profile must
average over all of it. The LDS lets the profile **evolve day to day** while
still shrinking neighbouring days together through $Q$. And because the model is
linear-Gaussian, $z^{\text{LDS}}$ can be **integrated out in closed form** by the
Kalman filter — that exact marginal is what makes the $C{=}0$ evidence exact and
the $C{=}1$ evidence (§5.1) tractable.

**Code.** Kalman recursions
[`kalman_filter` / `rts_smooth` / `kalman_sample` / `kalman_logpdf`](../models/graphical_model/non_ev_lds.py);
Gibbs adapter [`sample_z_lds`](../models/graphical_model/non_ev_lds.py).

### 2.2 $A, C, Q, R, \mu_0, \Sigma_0$ — global LDS parameters

**Distribution.** Global point-estimated parameters. Shapes
$A,Q,\Sigma_0\in\mathbb{R}^{L\times L}$, $C\in\mathbb{R}^{T\times L}$,
$R\in\mathbb{R}^{T\times T}$, $\mu_0\in\mathbb{R}^L$. Current setup: $A=C=I$ and
$Q,R,\Sigma_0$ diagonal.

**Fit.** **EM** over all training homes' $x^{\text{Non-EV}}$ sequences
([`fit_lds_em`](../models/graphical_model/non_ev_lds.py)). Each E-step runs the
Kalman smoother per home and accumulates Gaussian sufficient statistics; each
M-step is the closed-form LDS update. By default $A,C$ are **held at identity**
(`fit_A=fit_C=False`) and only $\mu_0,\Sigma_0,Q,R$ are updated, then projected
to diagonal; flags `fit_A` / `fit_C` / `fit_R` relax this without code changes.
Convergence: $|\Delta\log L|/|\log L| < 10^{-4}$ or 50 iters.

**Inference.** Read-only — the fixed LDS used by every Kalman pass above.

**Why EM (and why hold $A=C=I$)?** With $x^{\text{Non-EV}}$ observed in training
the smoother is exact, so EM climbs quickly to the MLE. Holding $A=C=I$ keeps the
latent **interpretable** (it is literally the per-day load profile) and the model
identifiable at $L=T=96$, while diagonal $Q,R$ preserve the factorize-across-$t$
structure the HMM relies on. Learning full $A,C$ is available but has not been
needed.

**Code.** [`fit_lds_em`, `_em_e_step`, `_em_m_step`](../models/graphical_model/non_ev_lds.py).

### 2.3 Coupling to the EV side (per-cell observation noise)

The EV and Non-EV submodels meet only through the **shared observation** $x$ and
the **per-cell variance** $\sigma^2_{k,t} = (\sigma^{\text{EV}}_k)^2 + R_{tt}$.
At inference this is implemented by passing the per-cell EV variance
$(\sigma^{\text{EV}}_{z_{d,t}})^2$ as an *extra additive observation covariance*
on top of $R$ in the Kalman pass — the `extra_obs_cov` argument of
[`kalman_filter`](../models/graphical_model/non_ev_lds.py). The latent
decomposition $x = x^{\text{EV}} + x^{\text{Non-EV}}$ is **never sampled**; it
stays marginalized throughout (the additive-Gaussian structure makes this free).

### 2.4 [Deprecated] Hierarchical-profile PPCA model

> **Deprecated.** Superseded by the LDS (§2.1–§2.3); removed from code, kept as
> contrast. It modelled each home's Non-EV days as i.i.d. around a *single*
> per-home mean profile — no across-day dynamics.

$$x^{\text{Non-EV},(n)}_{d,t} \stackrel{\text{iid over }d}{\sim} \mathcal{N}\!\left(\eta^{(n)}_t,\ (\omega^{(n)}_t)^2\right),\qquad \eta^{(n)}\sim\mathcal{N}\!\left(\bar\eta,\ WW^\top + \mathrm{diag}(\psi)\right).$$

The per-home mean profile $\eta^{(n)}\in\mathbb{R}^T$ carried a probabilistic-PCA
(factor-analyzer) cross-home prior — a global mean $\bar\eta$ plus a low-rank +
diagonal covariance fit from the $N$ training profiles — and the std profile
$\omega^{(n)}_t$ was either pooled-global or per-home with an Inverse-Gamma prior
(slice-sampled). Inference Gibbs-sampled $\eta^{(n)}$ as a $T$-dim Gaussian
conditional, integrating the $x$-decomposition out via the same
combined-variance trick as §2.3. **Why dropped:** a static per-home profile
cannot track day-to-day drift, which the LDS captures while staying
linear-Gaussian (hence Kalman-integrable). The PPCA prior's job — shrinking home
shapes toward plausible directions — is played in the LDS by $\Sigma_0$ and $Q$.

### 2.5 [Deprecated] Rank-1 scale-shape model

> **Deprecated.** The *original* Non-EV submodel. Every home shared a single
> global intraday shape $\rho_t$ ($\|\rho\|_2{=}1$), scaled by one per-home scalar.

$$x^{\text{Non-EV},(n)}_{d,t} \stackrel{\text{iid over }d}{\sim} \mathcal{N}\!\left(\alpha^{(n)}\rho_t,\ (\sigma^{\text{Non-EV}}_t)^2\right),\qquad \alpha^{(n)}\sim\mathcal{N}(\mu_\alpha,\ \sigma_\alpha^2).$$

$\rho$ was the top right singular vector of the stacked day-mean matrix; the
per-home scale $\alpha^{(n)}$ was Gibbs-sampled by Gaussian regression onto
$\rho$ with heteroscedastic-in-$z$ noise; the noise profile
$\sigma^{\text{Non-EV}}_t$ was pooled across homes. **Why dropped:** one global
shape for every household is far too rigid — superseded first by per-home PPCA
profiles (§2.4), then by the LDS (§2.1).

---

## 3. Total observation

### 3.1 $x^{(n)}_{d,t}$ — total grid power

**Definition.** $x^{(n)}_{d,t} = x^{\text{EV},(n)}_{d,t} + x^{\text{Non-EV},(n)}_{d,t}$
with independent Gaussian summands, so the marginal emission used at inference
is, **under the current LDS model:**

$$x^{(n)}_{d,t} \mid z^{(n)}_{d,t}{=}k,\ z^{\text{LDS},(n)}_d \sim \mathcal{N}\!\left(\Theta^{(n)}_k + (C\,z^{\text{LDS},(n)}_d)_t,\ \sigma^2_{k,t}\right),\quad \sigma^2_{k,t} = (\sigma^{\text{EV}}_k)^2 + R_{tt}.$$

(Under the deprecated submodels the Non-EV mean offset becomes $\eta^{(n)}_t$
(PPCA, §2.4) or $\alpha^{(n)}\rho_t$ (rank-1, §2.5), and $R_{tt}$ becomes
$(\omega^{(n)}_t)^2$ or $(\sigma^{\text{Non-EV}}_t)^2$.)

**Fit.** Fully observed in training; both components $x^{\text{EV}}$,
$x^{\text{Non-EV}}$ are also separately observed (training is on labeled data).

**Inference.** The *only* observed variable. Drives the FFBS emission
likelihoods in Block A and the data terms in Blocks B–C (and the exact $C{=}0$
smoother). Crucially, the latent decomposition
$x = x^{\text{EV}} + x^{\text{Non-EV}}$ is **not sampled** — it stays
marginalized throughout (see §2.3).

**Code.** Total-power arrays are assembled per home by
[`build_home_arrays()`](../models/graphical_model/_data.py); used throughout
[`infer_home_c0()`](../models/graphical_model/inference.py) and
[`infer_home_c1()`](../models/graphical_model/inference.py).

---

## 4. Inference loop (two-track)

The two EV hypotheses **share no parameters or latents**, so there is no reason
to alternate between them in a single chain (the old collapsed-$C$ Gibbs did;
see §4.4). Instead each home is inferred under both hypotheses *separately* and
the two are compared by their evidence (§5). Both tracks consume only
$x^{(n)}_{d,t}$ at inference time. Implemented in
[`models/graphical_model/inference.py`](../models/graphical_model/inference.py).

### 4.1 Track C = 0 (no EV) — exact, no sampling

Under $C=0$, $z^{(n)}_{d,t}\equiv\texttt{off}$ and $\Theta$ is irrelevant, so
the model collapses to the Non-EV daily LDS (§2). The only latent is
$z^{\text{LDS}}_{1:D}$, whose posterior is Gaussian and **exact**: a single
Kalman smoother yields its mean/covariance, and the Kalman filter's marginal
likelihood gives $\log p(x \mid C{=}0)$ with $z^{\text{LDS}}$ integrated out.
The off-state EV variance $\sigma^{\text{EV}}_{\texttt{off}}{}^2$ is added to
$\operatorname{diag}(R)$ so the expression matches the $C=1$ likelihood at
$z^{EV}\equiv\texttt{off}$. [`infer_home_c0()`](../models/graphical_model/inference.py).

### 4.2 Track C = 1 (EV) — three-block Gibbs

With $C=1$ fixed, $z^{EV}$ is free and we run a three-block Gibbs over
$(\Theta, z^{EV}, z^{\text{LDS}})$. Latent components
$x^{\text{EV}}, x^{\text{Non-EV}}$ remain *marginalized throughout* (§2.5).

1. **Block A — $z^{(n)}_{d,t}$** (§1.3) via HMM forward-filter backward-sample.
   Combined-emission likelihood
   $\mathcal{N}(\Theta^{(n)}_k + (C z^{\text{LDS}}_d)_t,\ (\sigma^{\text{EV}}_k)^2 + R_{tt})$,
   factorizing across $t$ given the current $z^{\text{LDS}}$.
2. **Block B — $\Theta^{(n)}_k$** for $k\in\{\texttt{low},\texttt{high}\}$ (§1.5).
   Truncated-Normal conjugate on residuals
   $x^{(n)}_{d,t} - (C z^{\text{LDS}}_d)_t$ over $(d,t)\in\mathcal{T}_k$, with
   heteroscedastic variance $\sigma^{\text{EV}}_k{}^2 + R_{tt}$.
3. **Block C — $z^{\text{LDS}}_{1:D}$** (§2) via Kalman FFBS on the residual
   $x - \Theta_{z^{EV}}$, with per-cell extra observation noise
   $\sigma^{\text{EV}}_{z_{d,t}}$ added to $\operatorname{diag}(R)$.

**Initialization.** $z^{\text{LDS}}$ = Kalman-smoother mean of $x$ (treats $x$
as all-Non-EV), $\Theta_k = \mu_{\Theta_k}$, $z^{EV}\equiv\texttt{off}$. Always
cold-starts. [`infer_home_c1()`](../models/graphical_model/inference.py).

**Schedule.** Default $S_{\text{burn}}=200$ burn-in + $S=500$ retained iters.

**Accumulation.** $z^{EV}$-counts incrementally post-burn-in; per-iter
$\Theta^{(n)}\in\mathbb{R}^K$ retained, and the predictive Non-EV mean
$\mathbb{E}[C z^{\text{LDS}}_d]$ accumulated (not stored per-iter). Optionally
the full $z^{EV}$ samples are retained (`retain_z_ev=True`) for the Chib
estimator (§5, B).

**Computational budget.** FFBS over $z^{EV}$ is $O(K^2 T D)$; block B is
$O(DT)$; block C (Kalman FFBS over days, each step a $T$-dim Gaussian) is
$O(D\,T^3)$ with $T^3\approx 10^6$ at $T{=}96$ — the dominant cost. The exact
$C=0$ smoother is one such Kalman pass. Total runtime stays under a few minutes
for a handful of homes.

### 4.3 Drivers

[`infer_home()`](../models/graphical_model/inference.py) runs both tracks for one
home and compares them; [`infer_all()`](../models/graphical_model/inference.py)
maps it over homes. [`build_heuristic_homes()`](../models/graphical_model/inference.py)
bridges the heuristic detector in as a comparison baseline.

### 4.4 [Deprecated] Collapsed-$C$ Gibbs (contrast only)

The previous sampler alternated a $C$-block into a single chain: each iteration
drew $C$ from its exact Bernoulli posterior
$p(C \mid x, z^{\text{LDS}}, \Theta)$ (HMM forward pass for the $C=1$ marginal
vs the $z\equiv\texttt{off}$ likelihood for $C=0$, weighted by $p_C$), then
$z^{EV}\mid C$. This conflated model selection with state inference and mixed
poorly once $C$ committed to a mode (the chain rarely flipped back). Removed in
favour of the two independent tracks above + an explicit evidence comparison.
Even older still: the rank-1 model's $z\to\Theta\to\alpha$ blocks (§2.5).

---

## 5. Evaluation

Implemented in [`evaluate()`](../models/graphical_model/evaluation.py) and
reported by [`print_evaluation()`](../models/graphical_model/evaluation.py).
Two confusion matrices:

- **EV ownership.** $2\times 2$ confusion of $\hat C^{(n)}$ vs true $C^{(n)}$
  across all evaluation homes
  ([`_c_confusion_from_probs()`](../models/graphical_model/evaluation.py)).
- **Charging state.** $3\times 3$ confusion of $\hat z^{(n)}_{d,t}$ vs true
  $z^{(n)}_{d,t}$ over $(n,d,t)$ on EV homes — per-home **row-normalised
  recall**, then averaged over homes so every home counts equally (hard:
  [`_per_home_z_confusion_hard()`](../models/graphical_model/evaluation.py);
  soft: [`_per_home_z_confusion_soft()`](../models/graphical_model/evaluation.py)).

For comparison, the heuristic's own per-timestep state output
([`first_diff_logistic.predict`](../models/first_diff_logistic.py)) is also
evaluated against the same ground truth — the baseline against which the Gibbs
sampler is compared. The inference notebook's presentation plots (confusion
matrices, $z^{\text{EV}}$ carpet heatmaps, charging-magnitude and
model-evidence figures) live in
[`notebooks/utils/infer_plots.py`](../notebooks/utils/infer_plots.py).

### 5.1 EV-ownership decision via model evidence

This section explains, from the ground up, **how we turn each home's data into a
yes/no EV decision and a probability** — the part the project leans on most.

#### The quantity we want

We never sample $C$. Instead we score the two hypotheses by their **model
evidence** — the probability the hypothesis assigns to the data — and pick the
larger:

$$\log p(x, C{=}c) \;=\; \underbrace{\log p(C{=}c)}_{\text{prior}} \;+\; \underbrace{\log p(x \mid C{=}c)}_{\text{marginal likelihood}}, \qquad c\in\{0,1\}.$$

The prior is just $p(C{=}1)=p_C$, $p(C{=}0)=1-p_C$. The hard part is the
**marginal likelihood** $\log p(x\mid C{=}c)$: the probability of the observed
total power $x$ under hypothesis $c$, *with all that hypothesis's latent
variables integrated (summed/averaged) out*. "Integrated out" is what makes it a
fair contest — each hypothesis is judged on how well it explains $x$ on its own
terms, not at one lucky setting of its latents.

The final decision and its soft version are
$$\hat C = \mathbb{1}\!\left[\log p(x, C{=}1) > \log p(x, C{=}0)\right], \qquad \hat P(C{=}1\mid x) = \operatorname{softmax}\big(\log p(x,C{=}0),\ \log p(x,C{=}1)\big)$$
([`HomeResult.c_prob`](../models/graphical_model/params.py)).

#### The easy side: $\log p(x \mid C{=}0)$ is exact

Under $C{=}0$ the only latent is $z^{\text{LDS}}$, and *everything is Gaussian*.
A Gaussian latent can be integrated out in closed form — that is precisely what
the **Kalman filter** computes as a by-product: $\log p(x\mid C{=}0) = \sum_d
\log p(x_d \mid x_{1:d-1})$, with $z^{\text{LDS}}$ already integrated. So the
$C{=}0$ evidence is **exact**, no sampling
([`lds_loglik_c0`](../models/graphical_model/inference.py); the off-state EV
variance $\sigma^{\text{EV}}_{\texttt{off}}{}^2$ is added to $\mathrm{diag}(R)$
so it lines up with the $C{=}1$ expression at $z^{\text{EV}}\equiv\texttt{off}$).

#### The hard side: $\log p(x \mid C{=}1)$ is intractable

Under $C{=}1$ the latents are $\psi = (z^{\text{EV}}, \Theta, z^{\text{LDS}})$,
and getting the marginal likelihood means summing over **every** charging-state
configuration $z^{\text{EV}}$ — that is $3^{D\times T}$ discrete paths — while
also integrating $\Theta$ and $z^{\text{LDS}}$. There is no closed form. We offer
three estimators, from cheap-and-rough to exact-but-slow.

**(A) Plug-in joint — cheap, but NOT comparable to $C{=}0$.** Evaluate the
complete-data joint density once, at a representative posterior point
$\psi^* = (z^{\text{EV}*}, \Theta^*, z^{\text{LDS}*})$ ($z^{\text{EV}*}=$ the MAP
of the per-cell marginals, $\Theta^*=$ posterior mean, $z^{\text{LDS}*}=$ its
conditional smoother mean):
$$\log p(C{=}1) + \log p\big(x, z^{\text{EV}*}, z^{\text{LDS}*}\mid C{=}1, \Theta^*\big).$$
The catch: this is a density over $(x, z^{\text{EV}}, z^{\text{LDS}})$ — it
includes the $\sim\!(D\times L)$-dimensional $\log p(z^{\text{LDS}*})$ prior
term — so it lives in a **much higher-dimensional space** than
$\log p(x\mid C{=}0)$ (a density over $x$ alone). Its absolute value is not on
the same scale, so it **must not be compared** to the $C{=}0$ evidence.
([`log_joint_c1_plugin`](../models/graphical_model/inference.py).)

**(A′) $z^{\text{LDS}}$-marginal — the default decision (`decision='rb'`).**
Keep the discrete $z^{\text{EV}}$ plugged in at $z^{\text{EV}*}$, but **integrate
$z^{\text{LDS}}$ out exactly** with the same Kalman trick as the $C{=}0$ side:
$$\log p(C{=}1) + \underbrace{\log p(z^{\text{EV}*}\mid C{=}1)}_{\text{HMM prior of the path}} + \underbrace{\log p(x\mid z^{\text{EV}*}, \Theta^*, C{=}1)}_{\text{Kalman marginal of the residual}}.$$
The second term is the HMM prior probability of the chosen state path; the third
runs the Kalman filter on the residual $x - \Theta^*[z^{\text{EV}*}]$ (with the
per-cell EV noise added to $R$) so $z^{\text{LDS}}$ is gone. The result is a
**density over $x$ alone**, hence directly comparable to $\log p(x\mid C{=}0)$.
It is still approximate — it fixes $z^{\text{EV}}$ and $\Theta$ at point
estimates instead of integrating them — but it is nearly free (one extra Kalman
pass) and is exactly the first two terms of the exact estimator below.
([`lds_loglik_c1_given_zev`](../models/graphical_model/inference.py).)

**(B) Chib (1995) — the exact marginal (up to Monte-Carlo error).** This removes
A′'s approximation by no longer fixing $z^{\text{EV}}$ and $\Theta$. It rests on
one rearrangement of Bayes' rule that holds **at any single point $\psi^*$**:

$$p(x\mid C{=}1) = \frac{p(x\mid\psi^*)\,p(\psi^*)}{p(\psi^*\mid x)} \quad\Longrightarrow\quad \log p(x\mid C{=}1) = \underbrace{\log p(x\mid\psi^*)}_{\text{likelihood}} + \underbrace{\log p(\psi^*)}_{\text{prior}} - \underbrace{\log p(\psi^*\mid x)}_{\text{posterior ordinate}}.$$

Read it as: *evidence = likelihood × prior ÷ posterior-density-at-that-point.*
The likelihood and prior at $\psi^*$ are easy to evaluate directly. The only
unknown is the **posterior ordinate** $p(\psi^*\mid x)$ — the height of the
posterior density at $\psi^*$ — which we don't have in closed form. Chib's idea:
**factor it along the Gibbs blocks** and estimate each factor from samples:

$$\log p(\psi^*\mid x) = \underbrace{\log p(z^{\text{EV}*}\mid x)}_{\text{ord}_{z^{\text{EV}}}} + \underbrace{\log p(\Theta^*\mid x, z^{\text{EV}*})}_{\text{ord}_{\Theta}} + \underbrace{\log p(z^{\text{LDS}*}\mid x, z^{\text{EV}*}, \Theta^*)}_{\text{ord}_{z^{\text{LDS}}}}.$$

Each factor conditions on the blocks **before** it (the Gibbs order
$z^{\text{EV}}\!\to\!\Theta\!\to\!z^{\text{LDS}}$) and is estimated differently:

- **$\text{ord}_{z^{\text{EV}}} = \log p(z^{\text{EV}*}\mid x)$.** We can compute
  the *full conditional* $p(z^{\text{EV}*}\mid x, \Theta, z^{\text{LDS}})$ exactly
  (HMM forward pass), so by Rao–Blackwell we **average that conditional over
  draws $(\Theta, z^{\text{LDS}})$ from the full posterior** — i.e. over the main
  Gibbs run (we keep a strided subsample of its draws). Averaging in probability
  space, in logs, is `logmeanexp`.
- **$\text{ord}_{\Theta} = \log p(\Theta^*\mid x, z^{\text{EV}*})$.** Now
  $z^{\text{EV}}$ is **pinned** at $z^{\text{EV}*}$. We run a short **reduced
  Gibbs** (only $\Theta$ and $z^{\text{LDS}}$ move) and average the
  truncated-Normal full-conditional density of $\Theta^*$ over those draws.
- **$\text{ord}_{z^{\text{LDS}}} = \log p(z^{\text{LDS}*}\mid x, z^{\text{EV}*},
  \Theta^*)$.** With both other blocks fixed this is just a Gaussian — the FFBS
  smoothing density — evaluated in **closed form**, no sampling
  ([`kalman_logpdf`](../models/graphical_model/non_ev_lds.py)).

Plugging the three back in gives the exact $\log p(x\mid C{=}1)$, then add
$\log p_C$ for the evidence. **Built-in sanity check.** The last two pieces obey
an exact Gaussian identity with no Monte-Carlo error:
$\log p(x\mid\psi^*) + \log p(z^{\text{LDS}*}) - \text{ord}_{z^{\text{LDS}}}$ must
reproduce A′'s Kalman-marginal term $\log p(x\mid z^{\text{EV}*}, \Theta^*,
C{=}1)$. The code computes both and reports their difference $\Delta$, which
should be $\approx 0$ — a free correctness test on the Kalman/prior/emission
plumbing. ([`chib_marginal_loglik_c1()`](../models/graphical_model/inference.py);
enable via `infer_home(..., decision='chib')` or
`infer_home_c1(..., compute_chib=True)`.)

**In one line.** $C{=}0$ evidence is exact; for $C{=}1$ use **A′** by default
(comparable, nearly free), and **B/Chib** when you want the exact marginal. The
density/evidence helpers (`lds_loglik_c0`, `lds_loglik_c1_given_zev`,
`log_hmm_prior`, `log_lds_prior`, `log_theta_prior`, `log_joint_c1_plugin`) are
reusable building blocks in
[`inference.py`](../models/graphical_model/inference.py).

---

## 6. Parameter summary table

### 6.1 Shared (EV side and total)

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $C^{(n)}$ | per-home latent | observed | evidence comparison $\log p(x,C{=}1)$ vs $\log p(x,C{=}0)$ (§5.1) | [`infer_home()`](../models/graphical_model/inference.py) |
| $p_C$ | global scalar | empirical mean | $C$ prior in the evidence (§5.1) | [`fit()`](../models/graphical_model/fit.py) |
| $z^{(n)}_{d,t}$ | per-home latent | observed | FFBS (C=1 Block A); $\equiv\texttt{off}$ if C=0 | [`ev.ffbs()`](../models/graphical_model/ev.py) |
| $\pi_z, P_z$ | global | smoothed counts | read-only | [`ev.fit_hmm()`](../models/graphical_model/ev.py) |
| $\Theta^{(n)}_k$ | per-home latent (truncated to $B_k$) | observed | C=1 Block B (truncated Gaussian) | [`ev.sample_theta_k()`](../models/graphical_model/ev.py) |
| $\mu_{\Theta_k}, \sigma_{\Theta_k}, \sigma^{\text{EV}}_k$ | global | EM | read-only | [`ev.fit_charging_em()`](../models/graphical_model/ev.py) |
| $x^{(n)}_{d,t}$ | observed | observed | observed | [`build_home_arrays()`](../models/graphical_model/_data.py) |

### 6.2 Non-EV side — current implementation (LDS)

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $z^{\text{LDS},(n)}_{1:D}$ | per-home latent ($D\times L$) | not inferred at fit | C=0: exact RTS smoother · C=1: Kalman FFBS (Block C) | [`sample_z_lds`](../models/graphical_model/non_ev_lds.py), [`rts_smooth`](../models/graphical_model/non_ev_lds.py) |
| $A, C$ | global ($L\times L$, $T\times L$) | EM (held at $I$ by default) | read-only | [`fit_lds_em`](../models/graphical_model/non_ev_lds.py) |
| $Q, R, \mu_0, \Sigma_0$ | global | EM (closed-form M-step, projected diagonal) | read-only | [`fit_lds_em`](../models/graphical_model/non_ev_lds.py) |

### 6.3 Non-EV side — deprecated predecessors (removed from code)

| Symbol | Kind | Fit | Model | Status |
|---|---|---|---|---|
| $\eta^{(n)}_t$; $\bar\eta, W, \psi$ | per-home $T$-vec; global PPCA prior | empirical day-mean; truncated-eigen FA | static per-home profile (PPCA, §2.4) | removed |
| $\omega^{(n)}_t$; $a^\omega_t, b^\omega_t$ | per-home $T$-vec; global IG prior | empirical per-$t$ var; method-of-moments | PPCA std profile (§2.4) | removed |
| $\alpha^{(n)}$; $\mu_\alpha, \sigma_\alpha^2$ | per-home scalar; global | plug-in OLS; mean / bias-corr. var | rank-1 scale (§2.5) | removed |
| $\rho_t$; $\sigma^{\text{Non-EV}}_t$ | global $T$-vec | top right SVD vector; pooled MSE | rank-1 shape / noise (§2.5) | removed |
