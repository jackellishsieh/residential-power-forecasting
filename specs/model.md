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

Each contribution is Gaussian conditional on per-home latents. Under the
**current/recommended** Non-EV submodel (§2.1–§2.6), the marginal observation
model used at inference is:

$$x^{(n)}_{d,t} \mid z^{(n)}_{d,t}{=}k \sim \mathcal{N}\!\left(\Theta^{(n)}_k + \eta^{(n)}_t,\ (\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2\right),$$

with the shorthand $\sigma^2_{k,t} \equiv (\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2$.
The **deprecated** rank-1 form (§2.7) replaces $\eta^{(n)}_t \to \alpha^{(n)}\rho_t$
and $(\omega^{(n)}_t)^2 \to (\sigma^{\text{Non-EV}}_t)^2$.

> **Implementation status.** The code in [`graphical_model.py`](../models/graphical_model.py)
> implements the hierarchical-profile form described in §2.1–§2.6.  The
> deprecated rank-1 submodel (§2.7) has been removed from the code and is
> kept here only as historical contrast.  Two parameterizations of the
> per-time variance profile $\omega$ are supported (selected by `omega_mode`
> in [`ModelParams`](../models/graphical_model.py#L100)); see §2.3.

### Phase summary

- **Fit** ([`fit()`](../models/graphical_model.py#L264)) uses fully-labeled
  training data — $C^{(n)}$, $z^{(n)}_{d,t}$, $x^{\text{EV}}$, $x^{\text{Non-EV}}$,
  and $x$ are all observed — to estimate the global parameter blocks in
  closed form (apart from a short EM loop for the charging-magnitude block
  and a small PPCA fit for the $\eta$ prior).  Output is a
  [`ModelParams`](../models/graphical_model.py#L100) dataclass.
- **Inference** has two drivers:
  - [`infer_all()`](../models/graphical_model.py#L1358) — mixture-Gibbs with
    a logistic-on-transitions heuristic step for $C$ on top of an exact
    marginal mixture step for $z$.  Legacy / debug.
  - [`infer_all_collapsed()`](../models/graphical_model.py#L1588) — clean
    collapsed Gibbs that samples $C$ from its exact marginal posterior
    (marginalising over $z$ via the HMM forward pass) and then $z$ from
    the backward conditional.  **Preferred default**; see §4.1.

  Both consume only $x^{(n)}_{d,t}$ at inference time and Gibbs-sample the
  per-home latents $(z, \Theta, \eta)$ — plus $\omega$ when `omega_mode='hierarchical'`.
- **Evaluation** ([`evaluate()`](../models/graphical_model.py#L1686),
  [`print_evaluation()`](../models/graphical_model.py#L1820)) compares
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
[`predict()`](../models/first_diff_logistic.py#L94)). Heuristic dispatch is
done by the caller; the model itself receives an initial $\hat C^{(n)}$ via
the `initial_c` argument to [`infer_home()`](../models/graphical_model.py#L781).

### 1.2 $p_C$ — EV prevalence

**Distribution.** Global scalar in $[0,1]$ (point-estimated).

**Fit.** Empirical mean $\hat p_C = \tfrac{1}{N}\sum_n C^{(n)}$. This is the
multinomial/Bernoulli MLE; no smoothing needed at this scale. Implemented in
[`fit()`](../models/graphical_model.py#L264).

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
[`infer_home()`](../models/graphical_model.py#L781) /
[`infer_home_collapsed()`](../models/graphical_model.py#L1413); the core
routines are [`_hmm_forward()`](../models/graphical_model.py#L1040),
[`_hmm_backward_sample()`](../models/graphical_model.py#L1086) and the
wrapper [`_ffbs()`](../models/graphical_model.py#L1108), with backward
draws delegated to
[`_sample_categorical_rows()`](../models/graphical_model.py#L1123). Cost is
$O(K^2 T D)$ per iter (\<1 ms in NumPy at $K{=}3, T{=}96, D{\le}360$).

**Posterior summary.** After burn-in, counts $z\text{-counts}[d,t,k]$ are
accumulated incrementally to avoid storing $S \cdot D \cdot T$ samples, then
normalized to per-cell marginals; the hard prediction is the argmax over $k$.

**Why FFBS over single-site Gibbs on $z_{d,t}$?** Adjacent states are strongly
coupled through $P_z$ (the chain is sticky in `off`); single-site Gibbs would
mix badly. FFBS is exact-conditional for HMMs and only marginally more code.

**Code.** [`_ffbs()`](../models/graphical_model.py#L1108),
[`_sample_categorical_rows()`](../models/graphical_model.py#L1123), called from
the Gibbs loop inside [`infer_home()`](../models/graphical_model.py#L781) and
[`infer_home_collapsed()`](../models/graphical_model.py#L1413).

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

**Code.** [`_fit_hmm()`](../models/graphical_model.py#L417).

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
[`graphical_model.py`](../models/graphical_model.py).

**Fit.** Observed in training (since $z$ and $x^{\text{EV}}$ are both labeled,
per-home means $\hat\theta^{(n)}_k = S_y^{(n)} / n^{(n)}_k$ are sufficient
stats). The *hyperparameters* $\mu_{\Theta_k},\sigma_{\Theta_k}^2,\sigma^{\text{EV}}_k$
are jointly fit by short EM — see §1.6. The truncation is not modeled in EM
(we treat $(\mu_{\Theta_k},\sigma_{\Theta_k}^2)$ as the parameters of the
*underlying* untruncated Normal). Since labeled $\hat\theta^{(n)}_k$ values
lie inside $B_k$ by construction of the labels, the bias from omitting the
truncation normalizer in the M-step is small; verbose mode prints the
fraction of *prior* probability mass inside $B_k$ as a sanity check.

**Inference.** Gibbs block 2 inside [`infer_home()`](../models/graphical_model.py#L781).
Conditional on current $z,\eta,\omega$, observations in state $k$ are
Gaussian with known mean offset and known per-time variance, giving a
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

**Code.** [`_sample_theta_k()`](../models/graphical_model.py#L1169),
[`_truncnorm_sample()`](../models/graphical_model.py#L1206).

### 1.6 $\mu_{\Theta_k}, \sigma_{\Theta_k}, \sigma^{\text{EV}}_k$ — charging-magnitude hyperparameters

**Distribution.** Global point-estimated parameters. $\sigma^{\text{EV}}_k$ is
the within-state, per-timestep emission std around $\Theta^{(n)}_k$. Off-state
is fixed: $\sigma^{\text{EV}}_{\texttt{off}}=10^{-3}$ (small floor for FFBS
numerical stability).

**Fit.** **EM** on the one-way Gaussian random-effects model
([`_fit_charging_em()`](../models/graphical_model.py#L665)). Initialized from
unbalanced ANOVA, then iterated to convergence ($|\Delta\log L|<10^{-6}$ or
100 iters). The E-step computes posterior moments
$\mathbb{E}[\Theta^{(n)}_k], \mathrm{Var}[\Theta^{(n)}_k]$ given current
hyperparameters; the M-step is closed form. Marginal log-likelihood is
monitored via [`_charging_loglik()`](../models/graphical_model.py#L760) and
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

**Code.** [`_fit_charging_em()`](../models/graphical_model.py#L665),
[`_charging_loglik()`](../models/graphical_model.py#L760).

---

## 2. Non-EV (background) submodel

> §2.1–§2.6 describe the **implemented** hierarchical per-home profile
> model.  §2.7 is a historical appendix preserving the deprecated rank-1
> scale-shape model (removed from code) as contrast.

**Per-day, per-home Non-EV emission.** For each home $n$, days are
conditionally i.i.d. given the per-home profiles:

$$x^{\text{Non-EV},(n)}_{d,t} \stackrel{\text{iid (over }d\text{)}}{\sim} \mathcal{N}\!\left(\eta^{(n)}_t,\ (\omega^{(n)}_t)^2\right).$$

Where the rank-1 model used a *global* shape $\rho_t$ and *global* noise
$\sigma^{\text{Non-EV}}_t$ modulated only by a per-home scalar $\alpha^{(n)}$,
the new model gives **every home its own $T$-vector mean profile and
$T$-vector std-dev profile**, with cross-home shrinkage supplied by
hierarchical priors. Variable names are chosen distinctively to avoid
collision with other model components: $\eta^{(n)}_t$ for the *mean* profile,
$\omega^{(n)}_t$ for the *std-dev* profile.

The covariance across $t$ within a home is **diagonal by construction** —
this is the property that lets the HMM's FFBS step factorize the emission
across $t$. Cross-time structure is captured only through the *prior* on
$\eta^{(n)}$ (§2.2), which is consulted in a separate Gibbs block where the
HMM is not involved.

### 2.1 $\eta^{(n)}_t$ — per-home Non-EV mean profile

**Distribution.**

$$\eta^{(n)} \stackrel{\text{iid}}{\sim} \mathcal{N}\!\left(\bar\eta,\ \Sigma_\eta\right),\qquad \Sigma_\eta = WW^\top + \mathrm{diag}(\psi),$$

with $\bar\eta\in\mathbb{R}^T$ a global mean profile, $W\in\mathbb{R}^{T\times r}$
a low-rank factor matrix, and $\psi\in\mathbb{R}^T_{>0}$ a per-time residual
variance. This is a **probabilistic-PCA / factor-analyzer prior** on
per-home shapes.  Rank $r$ is a knob: the code's
[`PPCA_RANK_DEFAULT`](../models/graphical_model.py#L85) is 5, but recent
experiments use $r{=}20$ (effectively saturating the available rank,
$\le N{-}1 = 49$) — the truncated-eigen fit (§2.2) is well-conditioned even
near full rank because of the diagonal residual $\psi$.

**Fit.** Per-home empirical day-mean profile from labeled training data:

$$\hat\eta^{(n)}_t = \tfrac{1}{D^{(n)}}\sum_{d=1}^{D^{(n)}} x^{\text{Non-EV},(n)}_{d,t}.$$

At $D^{(n)} \approx 365$, this is sharp — within-home noise in $\hat\eta^{(n)}_t$
is $(\omega^{(n)}_t)^2/D^{(n)}$, an order or two below cross-home variation.

**Inference.** Gibbs block 3 (see §4). Operates directly on the observed total
$x^{(n)}_{d,t}$ using the marginal combined-variance likelihood
$x^{(n)}_{d,t} - \Theta^{(n)}_{z_{d,t}} \sim \mathcal{N}\!\left(\eta^{(n)}_t,\ \sigma^2_{z_{d,t},t}\right)$
where $\sigma^2_{k,t} = (\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2$. Conditional
on current $z, \Theta, \omega^{(n)}$, the likelihood factorizes per $t$ into
i.i.d. Gaussian observations of $\eta^{(n)}_t$ with **heteroscedastic
variances** (different $d$ at the same $t$ have different variances because
$z_{d,t}$ varies); combined with the PPCA prior, the posterior is

$$\eta^{(n)} \sim \mathcal{N}\!\left(\Lambda^{-1}\,h,\ \Lambda^{-1}\right),$$

$$\Lambda = \Sigma_\eta^{-1} + \mathrm{diag}(\lambda_t),\quad \lambda_t = \sum_d \tfrac{1}{\sigma^2_{z_{d,t}, t}},\quad h_t^{(\text{data})} = \sum_d \tfrac{x^{(n)}_{d,t} - \Theta^{(n)}_{z_{d,t}}}{\sigma^2_{z_{d,t}, t}},\quad h = \Sigma_\eta^{-1}\bar\eta + h^{(\text{data})}.$$

$\Sigma_\eta^{-1}$ is computed once per iter via the Woodbury identity
($O(T r^2)$); the $T \times T$ Cholesky of $\Lambda$ is $\sim 10^5$ flops at
$T{=}96$ — negligible.

This is the **marginal-likelihood update**: we never sample the latent
component $x^{\text{Non-EV}}_{d,t}$. The additive decomposition
$x = x^{\text{EV}} + x^{\text{Non-EV}}$ stays implicit, as in the rank-1
model — see §2.5 for why we resisted data augmentation here.

**Why hierarchical (and *why low-rank+diagonal*)?**

- *Why per-home shapes at all.* The rank-1 model forces every home to share
  the same time-of-day pattern $\rho_t$. Empirically, homes vary in lifestyle
  (early risers vs. WFH vs. night-shift), and one shape can't capture this.
- *Why a prior at all.* At inference, only total $x$ is observed; per-home
  shape must be inferred from data that may be contaminated by EV charging.
  The cross-home prior is what shrinks toward something plausible.
- *Why low-rank+diagonal rather than diagonal-only.* With $N{=}50$ homes and
  $T{=}96$ timesteps, the empirical cross-home shape covariance is
  **rank-deficient by construction** ($\mathrm{rank}\le N{-}1 = 49 < T$).
  A plain $T{\times}T$ sample-covariance prior isn't invertible. Diagonal
  $\Sigma_\eta = \mathrm{diag}(\tau_t^2)$ sidesteps this but throws away the
  genuine cross-time correlations across homes (mornings and evenings
  co-vary). PPCA $WW^\top + \mathrm{diag}(\psi)$ captures dominant
  cross-home shape modes in $W$ while remaining well-conditioned. **It also
  defends against EV absorption:** charging-shaped (narrow, evening-peaked)
  deviations from $\bar\eta$ are *off the principal subspace*, so the prior
  pays $\sim 1/\psi_t$ in precision to absorb them. Diagonal-only prior
  doesn't penalize the *shape* of a deviation, only pointwise magnitude.
- *Alternatives considered:*
  - **Low-rank-only basis** $\eta^{(n)} = \Phi c^{(n)}$ with $\Phi\in\mathbb{R}^{T\times J}$
    fixed (Fourier / spline). Equivalent to $\Sigma_\eta = WW^\top$ with no
    diagonal residual — a strictly stronger constraint (homes are confined
    to an exact $J$-dim subspace). Worth keeping as a fallback if EV
    absorption empirically swamps the PPCA prior. Same Gibbs structure, just
    sample $c^{(n)}\in\mathbb{R}^J$ instead of $\eta^{(n)}\in\mathbb{R}^T$.
  - **Ledoit-Wolf shrinkage** $\Sigma_\eta = \lambda\hat\Sigma + (1-\lambda)\mathrm{diag}(\hat\tau^2)$.
    Auto-tunes $\lambda$ from data; less interpretable but cheap.
  - **Banded / GP-Matérn covariance.** Imposes stationarity in $t$, which is
    not obviously true (morning variation ≠ afternoon variation patterns).
  - **Smoothness prior across $t$** (AR(1) / GP on $\eta^{(n)}$). Subsumed
    by appropriate choice of $\Sigma_\eta$.

**Risk to flag.** Even with the PPCA prior, the new model is *strictly more
flexible* than the rank-1 model on the Non-EV side. If real-data $\hat z$
recovery regresses (Tier-2 eval, §2.6), the diagnosis is most likely that
genuine non-EV shape variation does include "evening peak"-like directions
that overlap with charging signatures.

**Code.** Per-home plug-in $\hat\eta^{(n)}$ is computed inside
[`_fit_background()`](../models/graphical_model.py#L456); the prior fit is
[`_fit_eta_prior()`](../models/graphical_model.py#L542); the per-iter
conditional sampler is [`_sample_eta()`](../models/graphical_model.py#L1188)
with $\Sigma_\eta^{-1}$ pre-cached via Woodbury in
[`_compute_sigma_eta_inv()`](../models/graphical_model.py#L1173).

### 2.2 $\bar\eta_t, W, \psi$ — hyperparameters of the $\eta$ prior

**Distribution.** Global point-estimated parameters. Together they define
$\Sigma_\eta = WW^\top + \mathrm{diag}(\psi)$.

**Fit.** Two stages from labeled training data:

1. **Global mean profile.** $\bar\eta_t = \tfrac{1}{N}\sum_n \hat\eta^{(n)}_t$.
2. **Factor decomposition.** PPCA on centered per-home profiles
   $\{\hat\eta^{(n)} - \bar\eta\}_{n=1}^N$. Either (a) ML-PPCA via EM, or
   (b) eigendecomposition of the sample covariance with $W = U_r\Lambda_r^{1/2}$
   and $\psi$ chosen to absorb residual diagonal variance. Both close to the
   MLE at $N{=}50$. **Rank choice.** Default $r{=}5$; can raise to 10. Plain
   diagonal corresponds to $r{=}0$. Full rank requires $N>T$ — infeasible.

**Bias correction.** The per-home plug-in $\hat\eta^{(n)}_t$ has within-home
noise $(\omega^{(n)}_t)^2 / D^{(n)}$. Sample covariance of $\hat\eta^{(n)}$
across homes overestimates $\Sigma_\eta$ by
$\mathrm{diag}\!\left(\tfrac{1}{N}\sum_n (\omega^{(n)}_t)^2 / D^{(n)}\right)$.
Subtract this from $\psi$ (floor at a small positive value) to debias —
analogous to Step 3e of the deprecated fit.

**Inference.** Read-only — fixed prior parameters for the $\eta^{(n)}$
conditional (§2.1).

**Why MoM/EM-PPCA?** With $N{=}50$ samples in $T{=}96$ dimensions, this is the
small-$N$ regime where regularized estimators (factor analysis with chosen $r$,
or shrinkage) genuinely beat the empirical sample covariance. Off-the-shelf
PPCA-EM converges in tens of iterations.

**Code.** [`_fit_eta_prior()`](../models/graphical_model.py#L542) implements
the truncated-eigen variant (option b above), with bias correction on the
diagonal as described.

### 2.3 $\omega^{(n)}_t$ — Non-EV std-dev profile

**Two parameterizations are supported** (selected by `omega_mode` in `ModelParams`):

- **`omega_mode = "global"` (default).** A single $T$-vector $\sigma^{\text{Non-EV}}_t{}^2$ is
  fit pooled across all training homes at fit time and held **fixed** at
  inference. No Gibbs block. This is the same structure as the deprecated
  rank-1 model used for noise (§2.7.4) — only the *mean* side has changed
  to hierarchical PPCA.
- **`omega_mode = "hierarchical"`.** Per-home, per-$t$ variance, with IG prior
  detailed below. Sampled at inference via slice sampling.

**Why is `"global"` the default?** A diagnostic concern about the hierarchical
parameterization: at inference, the slice sampler can shrink $\omega^{(n)}_t$
*conditional on the current $z$*. If $z$ flips a timestep to a charging state
(rightly or wrongly), the residual under that $z$ is small, so the slice
sampler tightens $\omega[t]$. A tighter $\omega[t]$ makes the off-state
predictive narrower → smaller residuals look like charging → more flips →
$\omega$ shrinks further. The "global" parameterization breaks this feedback
loop by removing $\omega$ from inference entirely.

The hierarchical option is preserved both because it's more theoretically
expressive and as an A/B for diagnosing how much the feedback loop matters
in practice.

#### Hierarchical (opt-in) details

**Distribution.** Independently across $t$ and $n$:

$$(\omega^{(n)}_t)^2 \stackrel{\text{iid}}{\sim} \mathrm{InvGamma}(a^\omega_t, b^\omega_t).$$

This is the conjugate prior for an unknown variance under Gaussian observations.

**Fit.** Per-home, per-time empirical variance against the fitted mean:

$$\widehat{(\omega^{(n)}_t)^2} = \tfrac{1}{D^{(n)}}\sum_{d=1}^{D^{(n)}} \left(x^{\text{Non-EV},(n)}_{d,t} - \hat\eta^{(n)}_t\right)^2.$$

**Inference.** Gibbs block 4 (see §4). The posterior under the marginal
combined-variance likelihood is **not conjugate** — the unknown $(\omega^{(n)}_t)^2$
enters the likelihood through the sum $\sigma^{\text{EV}}_{z_{d,t}}^2 + (\omega^{(n)}_t)^2$,
not as the full variance. The log-posterior is, scalar per $t$ and
independent across $t$:

$$\log p\!\left((\omega^{(n)}_t)^2 \,\big|\, z, \Theta, \eta, x\right) \;=\; \log p_{\mathrm{IG}}\!\left((\omega^{(n)}_t)^2; a^\omega_t, b^\omega_t\right) \;-\; \tfrac{1}{2}\sum_d \!\left[\log\sigma^2_{z_{d,t}, t} + \tfrac{(x_{d,t} - \Theta_{z_{d,t}} - \eta^{(n)}_t)^2}{\sigma^2_{z_{d,t}, t}}\right]$$

with $\sigma^2_{k,t} = (\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2$. We sample
each $(\omega^{(n)}_t)^2$ with a **univariate slice sampler** (Neal 2003)
with stepping-out and shrinkage, working in log-space
($\ell = \log(\omega^{(n)}_t)^2$) for scale invariance. Slice sampling is
tuning-free, robust on unimodal targets, and well-suited to scalar
posteriors of this form.

**Why per-home, per-$t$?** The rank-1 model used a *globally pooled*
$\sigma^{\text{Non-EV}}_t$ — every home shares the same noise profile. This is
clearly wrong: homes differ in HVAC cycling, appliance usage, baseline
variability. We let homes have their own scale; the hierarchical IG prior
shares strength across homes.

**Why diagonal across $t$ (independent prior per $t$)?** Emission variance
enters the FFBS forward pass diagonally; the per-$t$ independence isn't
required for the slice sampler itself (each $t$ would still be a scalar
update) but it keeps the prior fit (§2.4) two-parameters-per-$t$ and avoids
specifying a $T$-dim joint distribution over variances.

**Why slice sampling, not conjugate IG with data augmentation?** Augmenting
the latent $x^{\text{EV}}, x^{\text{Non-EV}}$ decomposition would restore
conjugacy (the $\omega$ update becomes $\mathrm{IG}(a + D/2,\ b + \tfrac{1}{2}\sum_d (x^{\text{Non-EV}}_{d,t} - \eta_t)^2)$
exactly) — at the cost of (a) throwing away the *marginal-likelihood* property
that made the rank-1 model elegant, (b) introducing autocorrelation between
augmented latents and $\omega$ samples that empirically slows mixing for
variance components, and (c) carrying two $(D,T)$ latent arrays in
`HomeInference`. Slice sampling preserves marginalization and adds only
~30 lines of code. See §2.5.

**Alternatives considered.**
- **Conjugate IG with data augmentation.** See §2.5.
- **Plug-in MAP (Stochastic-EM)** of $(\omega^{(n)}_t)^2$ via 1-D Newton
  per iter. Faster than slice sampling but discards Bayesian uncertainty
  over $\omega$.
- **Log-Normal hierarchy.** More "natural" prior on a positive scale; loses
  conjugacy *and* doesn't help with the sum-variance non-conjugacy. Rejected.
- **Half-Cauchy on $\omega^{(n)}_t$** with a higher-level scale. Common in
  Bayesian hierarchical literature; same non-conjugacy story.

**Code.** Slice sampler in log-space:
[`_sample_omega()`](../models/graphical_model.py#L1237) and the generic
1-D slice routine
[`_slice_sample_1d()`](../models/graphical_model.py#L1297) (stepping-out +
shrinkage, capped iters).  Fit-time pooled variance for the global mode is
[`_fit_omega_global()`](../models/graphical_model.py#L516); fit-time
hierarchical prior is
[`_fit_omega_prior()`](../models/graphical_model.py#L622).

### 2.4 $a^\omega_t, b^\omega_t$ — hyperparameters of the $\omega$ prior

**Distribution.** Global Inverse-Gamma shape ($a^\omega_t$) and rate ($b^\omega_t$)
per timestep.

**Fit.** Method-of-moments on $\{\widehat{(\omega^{(n)}_t)^2}\}_{n=1}^N$ across
homes. Match the empirical mean and variance of the per-home variance
estimates to the IG mean $b/(a-1)$ and variance $b^2/((a-1)^2(a-2))$
(requires $a > 2$; floor if needed).

**Inference.** Read-only — fixed prior parameters for the $\omega^{(n)}_t$
conditional (§2.3).

**Why MoM?** Closed-form, two parameters per $t$, fit from $N{=}50$ samples.
Alternative: numerical MLE for the IG (no closed form, easy 1-D root-finding).
MoM is good enough at this $N$.

**Code.** [`_fit_omega_prior()`](../models/graphical_model.py#L622) (only
populated when `omega_mode='hierarchical'`; in `'global'` mode this is
skipped and `a_omega, b_omega = None`).

### 2.5 Why we keep $x^{\text{EV}}, x^{\text{Non-EV}}$ marginalized at inference

**The temptation: data augmentation.** Inverse-Gamma is conjugate to
$\mathcal{N}(\mu, \omega^2)$ but **not** to $\mathcal{N}(\mu, c + \omega^2)$
where $c = (\sigma^{\text{EV}}_k)^2$ is a known offset. A textbook workaround is
to sample the latent decomposition $x^{(n)}_{d,t} = x^{\text{EV},(n)}_{d,t} + x^{\text{Non-EV},(n)}_{d,t}$
as auxiliary variables — the Gaussian-sum conditional is closed form — and
then use the augmented $\{x^{\text{Non-EV}}_{d,t}\}$ to drive a fully
conjugate IG update on $(\omega^{(n)}_t)^2$.

**Why we don't do this.** The rank-1 model never decomposed $x$; every Gibbs
block (FFBS, $\Theta$, $\alpha$) operated on the *marginal* combined-variance
likelihood. This is structurally elegant *and* avoids a well-known mixing
pathology of data augmentation for variance components: the augmented
latents $x^{\text{Non-EV}}_{d,t}$ and the variance $\omega^{(n)}_t$ are
strongly correlated under the joint posterior, so each Gibbs step moves them
in lock-step, slowing chain mixing. The cost in raw flops/memory is small
($DT$ Gaussian draws + two $(D,T)$ arrays), but the mixing cost is the kind
that hides in 500-iter chains and only shows up when you check trace plots.

**What we do instead.** Slice-sample $(\omega^{(n)}_t)^2$ directly under the
marginal log-posterior (see §2.3 inference). The other blocks ($\Theta_k$,
$\eta$) stay as marginal-likelihood conjugate Gaussian updates with
heteroscedastic-in-$z$ variances — same skeleton as the rank-1 model.

**Cost of slice sampling.** $T$ scalar slice samples per home per iter; each
needs ~5–10 log-density evaluations, each $O(D)$. Total $O(\text{const}\cdot TD)$ per
iter — same big-O as augmentation, modestly larger constant. No latent
$x^{\text{EV}}, x^{\text{Non-EV}}$ arrays in `HomeInference`.

**Falling back to augmentation.** If slice mixing turns out to be poor in
practice (e.g., heavy-tailed posteriors at small $D^{(n)}$), the augmentation
path is documented above and ~50 lines of additional code; we can swap in
without changing other blocks.

### 2.6 Evaluation plan for the migration

Two tiers, in order:

**Tier 1 — synthetic recovery.** Simulate from the new generative model with
known $\eta^{(n)}, \omega^{(n)}, z, \Theta$ values drawn from priors. Run
inference end-to-end. Required to pass:
- $\hat z$ confusion matrix on synthetic data is **no worse than** the
  rank-1 model's $\hat z$ confusion matrix on *its own* synthetic data
  (apples to apples — each model evaluated against data from itself).
- $\hat\eta^{(n)}_t$ recovery MAE $\le$ $\sqrt{\psi_t}$ on average across
  homes. (We can't expect to beat the prior std when starting from the
  prior mean; we should at least come within it.)
- $\widehat{(\omega^{(n)}_t)^2}$ posterior mean MAE $\le$ prior std.

If Tier 1 fails: model is non-identifiable at $N{=}50, D{\approx}365$
even without model mismatch. Tighten prior (lower $r$, smaller $\psi$, or
revert to low-rank-only basis $\Sigma_\eta = WW^\top$).

**Tier 2 — real-data parity.** Same held-out evaluation homes, same metrics
already computed by [`evaluate()`](../models/graphical_model.py#L913):
$\hat C$ confusion (unchanged — still heuristic) and $\hat z$ confusion
(EV homes via Gibbs). Required: $\hat z$ confusion doesn't materially
regress vs. the rank-1 baseline.

If Tier 1 passes but Tier 2 regresses: real non-EV shapes have substantial
mass in directions the PPCA prior treats as cheap and overlap with charging.
Mitigations: (a) raise $r$ in PPCA (loosen prior — opposite direction;
probably wrong); (b) lower $r$ or impose hard basis (tighten prior); (c)
re-fit with explicit out-of-subspace penalty.

### 2.7 [Deprecated] Rank-1 scale-shape model

> **Deprecated.** This is the *original* Non-EV submodel and is the one
> currently implemented in code. It is kept here as a contrast to §2.1–§2.6.
> Migration to the hierarchical-profile form is pending.

The deprecated model factorizes the per-home Non-EV emission as a *global
shape times a per-home scalar scale*, with a *global per-time noise*:

$$x^{\text{Non-EV},(n)}_{d,t} \stackrel{\text{iid (over }d\text{)}}{\sim} \mathcal{N}\!\left(\alpha^{(n)}\rho_t,\ (\sigma^{\text{Non-EV}}_t)^2\right).$$

#### 2.7.1 $\alpha^{(n)}$ — per-home background scale (deprecated)

**Distribution.** $\alpha^{(n)} \stackrel{\text{iid}}{\sim} \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)$.

**Fit.** Plug-in OLS projection of the home's day-mean profile onto $\rho$:

$$\hat\alpha^{(n)} = \sum_t \hat\beta^{(n)}_t\,\rho_t, \quad \hat\beta^{(n)}_t = \tfrac{1}{D^{(n)}}\sum_d x^{\text{Non-EV},(n)}_{d,t}.$$

**Inference.** Gibbs block 3 inside [`infer_home()`](../models/graphical_model.py#L490):
Gaussian regression onto fixed regressor $\rho_t$ with heteroscedastic noise
$\sigma^2_{z_{d,t},t}$:

$$\alpha^{(n)} \sim \mathcal{N}(m_\alpha,\,1/\text{prec}_\alpha),\quad \text{prec}_\alpha = \tfrac{1}{\sigma_\alpha^2} + \sum_{d,t}\tfrac{\rho_t^2}{\sigma^2_{z_{d,t},t}}.$$

**Why deprecated.** Forces a single global shape across all homes — too
restrictive given real cross-home variation in load patterns.

**Code.** Fit: [`_fit_background()`](../models/graphical_model.py#L300).
Inference: [`_sample_alpha()`](../models/graphical_model.py#L789).

#### 2.7.2 $\mu_\alpha, \sigma_\alpha^2$ — cross-home scale hyperparameters (deprecated)

**Fit.** $\mu_\alpha = \tfrac{1}{N}\sum_n \hat\alpha^{(n)}$; $\sigma_\alpha^2$
with bias correction subtracting noise variance of the plug-in:

$$\sigma_\alpha^2 = \max\!\left(0,\ \mathrm{Var}_n(\hat\alpha^{(n)}) - \tfrac{1}{N}\sum_n \tfrac{1}{D^{(n)}}\sum_t \rho_t^2(\sigma^{\text{Non-EV}}_t)^2\right).$$

**Code.** [`_fit_background()`](../models/graphical_model.py#L300).

#### 2.7.3 $\rho_t$ — global normalized intraday background profile (deprecated)

**Distribution.** Global vector $\rho\in\mathbb{R}^T$ with $\|\rho\|_2=1$.

**Fit.** Top right singular vector of the stacked day-mean matrix
$B\in\mathbb{R}^{N\times T}$ via economy SVD. Sign-fixed so that
$\mathrm{median}_n(\hat\alpha^{(n)}) > 0$.

**Why deprecated.** A single $\rho$ is the entire rank-1 constraint — see
"why deprecated" under §2.7.1.

**Code.** [`_fit_background()`](../models/graphical_model.py#L300).

#### 2.7.4 $\sigma^{\text{Non-EV}}_t$ — per-time background emission noise (deprecated)

**Distribution.** Global vector in $\mathbb{R}^T_{>0}$ (pooled across homes).

**Fit.** Pooled residual variance against the per-home fitted mean,
**individual obs** (not day-means), $\sum_n D^{(n)} \cdot T \approx 1.5$M
data points:

$$(\sigma^{\text{Non-EV}}_t)^2 = \tfrac{1}{\sum_n D^{(n)}}\sum_n\sum_d \left(x^{\text{Non-EV},(n)}_{d,t} - \hat\alpha^{(n)}\rho_t\right)^2.$$

**Why deprecated.** Pooling forces all homes to share noise scale — wrong
empirically. Replaced by per-home $\omega^{(n)}_t$ (§2.3).

**Code.** [`_fit_background()`](../models/graphical_model.py#L300).

---

## 3. Total observation

### 3.1 $x^{(n)}_{d,t}$ — total grid power

**Definition.** $x^{(n)}_{d,t} = x^{\text{EV},(n)}_{d,t} + x^{\text{Non-EV},(n)}_{d,t}$
with independent Gaussian summands, so the marginal emission used at inference
is, **under the new model:**

$$x^{(n)}_{d,t} \mid z^{(n)}_{d,t}{=}k \sim \mathcal{N}\!\left(\Theta^{(n)}_k + \eta^{(n)}_t,\ \sigma^2_{k,t}\right),\quad \sigma^2_{k,t} = (\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2.$$

(Under the deprecated rank-1 model: $\eta^{(n)}_t \to \alpha^{(n)}\rho_t$ and
$(\omega^{(n)}_t)^2 \to (\sigma^{\text{Non-EV}}_t)^2$.)

**Fit.** Fully observed in training; both components $x^{\text{EV}}$,
$x^{\text{Non-EV}}$ are also separately observed (training is on labeled data).

**Inference.** The *only* observed variable. Drives the FFBS emission
likelihoods in block 1 and the data terms in blocks 2–4. Crucially, the
latent decomposition $x = x^{\text{EV}} + x^{\text{Non-EV}}$ is **not
sampled** — it stays marginalized throughout, exactly as in the rank-1
model (see §2.5).

**Code.** Total-power arrays are assembled per home by
[`_build_home_arrays()`](../models/graphical_model.py#L391); used throughout
[`infer_home()`](../models/graphical_model.py#L781) and
[`infer_home_collapsed()`](../models/graphical_model.py#L1413).

---

## 4. Inference loop (cross-cutting)

Per-home Gibbs sampler.  Two drivers are provided; both consume only
$x^{(n)}_{d,t}$ at inference time:

- [`infer_home()`](../models/graphical_model.py#L781): the original
  mixture-Gibbs sampler.  Each iteration does an exact marginal mixture step
  on $z$ (FFBS proposal weighted vs all-off by $p_C$) **and**, separately,
  a logistic-on-transitions step on $C$.  Retained for backward compat /
  diagnostics; the two $C$-related steps look at slightly different posteriors.
- [`infer_home_collapsed()`](../models/graphical_model.py#L1413): the clean
  collapsed Gibbs.  Each iteration samples $C$ from its exact marginal
  posterior $p(C \mid x, \Theta, \eta, \omega)$ (marginalising over $z$ via
  the HMM forward pass), then $z \mid C$ from the backward sample.  No
  logistic regression; **preferred default for new work**.

Both apply the per-home Gibbs blocks below; the only difference is the $C$
update.

### 4.1 Block structure (current implementation)

Each iteration executes the following blocks in order.  Latent components
$x^{\text{EV}}, x^{\text{Non-EV}}$ remain *marginalized throughout* (see §2.5).

1. **FFBS for $z^{(n)}_{d,t}$** (§1.3). Combined-emission likelihood
   $\mathcal{N}(\Theta^{(n)}_k + \eta^{(n)}_t,\ (\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2)$.
   Factorizes across $t$ given current parameters.
2. **$\Theta^{(n)}_k$** for $k\in\{\texttt{low},\texttt{high}\}$ (§1.5).
   Conjugate Gaussian on residuals $x^{(n)}_{d,t} - \eta^{(n)}_t$ over
   $(d,t)\in\mathcal{T}_k$, with heteroscedastic variance
   $\sigma^{\text{EV}}_k{}^2 + \omega^{(n)}_t{}^2$.
3. **$\eta^{(n)}$** (§2.1). $T$-dim Gaussian conjugate update on residuals
   $x^{(n)}_{d,t} - \Theta^{(n)}_{z_{d,t}}$ with per-cell heteroscedastic
   variance $\sigma^2_{z_{d,t}, t}$ and PPCA prior
   $\mathcal{N}(\bar\eta, \Sigma_\eta)$. $T\times T$ Cholesky; Woodbury for
   $\Sigma_\eta^{-1}$ (cached once per `infer_home` call since
   $\Sigma_\eta$ doesn't depend on iter state).
4. **$\omega^{(n)}_t$** (§2.3).  **Only executed when `omega_mode='hierarchical'`.**
   Univariate slice sample in log-space per $t$, on the marginal log-posterior
   with IG prior.  In `omega_mode='global'` (default) this block is skipped
   and $\omega^2_t = \sigma^{\text{Non-EV}}_t{}^2$ stays fixed for the chain.

**Initialization.** $\eta^{(n)} = \bar\eta$,
$\Theta^{(n)}_k = \mu_{\Theta_k}$, $z\equiv\texttt{off}$, and
$\omega^2_t = \sigma^{\text{Non-EV}}_t{}^2$ (global mode) or
$\omega^2_t = b^\omega_t / (a^\omega_t + 1)$ (hierarchical IG mode).
`infer_home` additionally accepts warm-start values for $(C, z)$ via the
`initial_c` / `initial_z` arguments (used to seed from the heuristic
detector); `infer_home_collapsed` always cold-starts.

**Schedule.** Default $S_{\text{burn}}=200$ burn-in + $S=500$ retained iters.

**Accumulation.** $z$-counts incrementally post-burn-in; per-iter
$\eta^{(n)} \in \mathbb{R}^T$, $\omega^{(n)} \in \mathbb{R}^T$, and
$\Theta^{(n)} \in \mathbb{R}^K$ samples retained (cheap; $S\cdot T$ and
$S\cdot K$ floats per home).

**Computational budget.** FFBS dominates ($O(K^2 T D)$); block 2 is
$O(\sum_k|\mathcal{T}_k|) = O(DT)$; block 3 is $O(DT + T^3 + Tr^2)$ with
$T^3 \approx 10^6$ at $T{=}96$; block 4 (only in hierarchical mode) is
$O(\text{slice-evals}\cdot TD)$ with typically 5–10 evals per slice.  Total
runtime stays under a few minutes for a handful of EV homes.

### 4.2 [Removed] Deprecated three-block Gibbs

The old rank-1 model used blocks $z \to \Theta \to \alpha$ (scalar regression
on $\alpha^{(n)}$).  No longer in the code; see §2.7 for the math, kept as
contrast only.

### 4.3 Cross-cutting

- **Log-likelihood tracking.** [`_compute_loglik()`](../models/graphical_model.py#L1000)
  for EV homes (complete-data likelihood under the hierarchical model);
  [`_compute_loglik_c0()`](../models/graphical_model.py#L1019) for the
  all-off branch used in the C-mixture step.  Both use the combined-variance
  emission $\mathcal{N}(\Theta_k + \eta_t, \sigma^2_{k,t})$.
- The drivers [`infer_all()`](../models/graphical_model.py#L1358) and
  [`infer_all_collapsed()`](../models/graphical_model.py#L1588) iterate
  over homes; [`c_prob_from_z_via_heuristic()`](../models/graphical_model.py#L1637)
  and [`build_heuristic_homes()`](../models/graphical_model.py#L1669) bridge
  the heuristic detector into the per-home inference pipeline.
- **Memory per home.** Dominated by $z$-counts of shape $(D, T, K)$ — under
  1 MB.  No latent $x^{\text{EV}}, x^{\text{Non-EV}}$ arrays are stored
  (marginalization is preserved; see §2.5).

---

## 5. Evaluation

Implemented in [`evaluate()`](../models/graphical_model.py#L1686) and reported
by [`print_evaluation()`](../models/graphical_model.py#L1820). Two confusion
matrices:

- **EV ownership.** $2\times 2$ confusion of $\hat C^{(n)}$ vs true $C^{(n)}$
  across all evaluation homes
  ([`_c_confusion_from_probs()`](../models/graphical_model.py#L1782)).
- **Charging state.** $3\times 3$ confusion of $\hat z^{(n)}_{d,t}$ vs true
  $z^{(n)}_{d,t}$ across all $(n,d,t)$ triples on EV homes (hard:
  [`_per_home_z_confusion_hard()`](../models/graphical_model.py#L1753); soft:
  [`_per_home_z_confusion_soft()`](../models/graphical_model.py#L1765);
  averaging: [`_nanmean_cms()`](../models/graphical_model.py#L1776)).

For comparison, the heuristic's own per-timestep state output
([`first_diff_logistic.predict`](../models/first_diff_logistic.py#L94)) is
also evaluated against the same ground truth — this is the baseline against
which the Gibbs sampler is compared.

---

## 6. Parameter summary table

### 6.1 Shared (EV side and total)

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $C^{(n)}$ | per-home latent | observed | collapsed marginal (default) / mixture | [`infer_home_collapsed()`](../models/graphical_model.py#L1413) / [`infer_home()`](../models/graphical_model.py#L781) |
| $p_C$ | global scalar | empirical mean | read-only | [`fit()`](../models/graphical_model.py#L264) |
| $z^{(n)}_{d,t}$ | per-home latent | observed | FFBS (block 1) | [`_ffbs()`](../models/graphical_model.py#L1108) |
| $\pi_z, P_z$ | global | smoothed counts | read-only | [`_fit_hmm()`](../models/graphical_model.py#L417) |
| $\Theta^{(n)}_k$ | per-home latent (truncated to $B_k$) | observed | Gibbs block 2 (truncated Gaussian) | [`_sample_theta_k()`](../models/graphical_model.py#L1169) |
| $\mu_{\Theta_k}, \sigma_{\Theta_k}, \sigma^{\text{EV}}_k$ | global | EM | read-only | [`_fit_charging_em()`](../models/graphical_model.py#L665) |
| $x^{(n)}_{d,t}$ | observed | observed | observed | [`_build_home_arrays()`](../models/graphical_model.py#L391) |

### 6.2 Non-EV side — current implementation

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $\eta^{(n)}_t$ | per-home latent ($T$-vec) | empirical day-mean | Gibbs block 3 ($T$-dim Gaussian, heteroscedastic) | [`_sample_eta()`](../models/graphical_model.py#L1188) |
| $\bar\eta_t, W, \psi$ | global ($\Sigma_\eta = WW^\top + \mathrm{diag}(\psi)$) | mean + truncated-eigen FA with bias-correction | read-only | [`_fit_eta_prior()`](../models/graphical_model.py#L542) |
| $\sigma^{\text{Non-EV}}_t$ *(when `omega_mode='global'`, default)* | global ($T$-vec) | pooled per-$t$ MSE | read-only (fixed) | [`_fit_omega_global()`](../models/graphical_model.py#L516) |
| $\omega^{(n)}_t$ *(when `omega_mode='hierarchical'`)* | per-home latent ($T$-vec) | empirical per-$t$ var | Gibbs block 4 (slice sample, per-$t$) | [`_sample_omega()`](../models/graphical_model.py#L1237) |
| $a^\omega_t, b^\omega_t$ *(when `omega_mode='hierarchical'`)* | global ($T$-vec each) | method-of-moments | read-only | [`_fit_omega_prior()`](../models/graphical_model.py#L622) |

### 6.3 Non-EV side — deprecated rank-1 (removed from code)

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $\alpha^{(n)}$ | per-home latent (scalar) | plug-in OLS | (was Gibbs block 3) | removed |
| $\mu_\alpha, \sigma_\alpha^2$ | global | mean / bias-corrected var | read-only | removed |
| $\rho_t$ | global ($T$-vec) | top right SVD vector | read-only | removed |
| $\sigma^{\text{Non-EV}}_t$ | global ($T$-vec) | individual-obs MSE | read-only | (preserved as the `omega_mode='global'` fit, see §6.2) |
