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
> currently implements the deprecated rank-1 form. The hierarchical-profile
> form documented in §2.1–§2.6 is the design we are migrating to and is not
> yet implemented. Code pointers in §2.1–§2.6 are therefore aspirational
> ("will live in …"); those in §2.7 reflect the actual current code.

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

> **Two versions documented.** Sections §2.1–§2.6 describe the
> **current/recommended** hierarchical per-home profile model that we are
> migrating to. Section §2.7 preserves the **deprecated** rank-1 scale-shape
> model for contrast (and because the current code still implements it).

**Per-day, per-home Non-EV emission (new model).** For each home $n$,
days are conditionally i.i.d. given the per-home profiles:

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
a low-rank factor matrix (rank $r \in [5, 10]$ recommended), and
$\psi\in\mathbb{R}^T_{>0}$ a per-time residual variance. This is a
**probabilistic-PCA / factor-analyzer prior** on per-home shapes.

**Fit.** Per-home empirical day-mean profile from labeled training data:

$$\hat\eta^{(n)}_t = \tfrac{1}{D^{(n)}}\sum_{d=1}^{D^{(n)}} x^{\text{Non-EV},(n)}_{d,t}.$$

At $D^{(n)} \approx 365$, this is sharp — within-home noise in $\hat\eta^{(n)}_t$
is $(\omega^{(n)}_t)^2/D^{(n)}$, an order or two below cross-home variation.

**Inference.** Gibbs block 4 (see §4). Conditional on the augmented latent
$\{x^{\text{Non-EV},(n)}_{d,t}\}$ (§2.5) and current $\omega^{(n)}$, the
likelihood factorizes per $t$ into i.i.d. Gaussian observations of $\eta^{(n)}_t$
with variance $(\omega^{(n)}_t)^2$. Combined with the $T$-dim Gaussian prior,
the conditional posterior is

$$\eta^{(n)} \sim \mathcal{N}\!\left(\Lambda^{-1}\,h,\ \Lambda^{-1}\right),$$

$$\Lambda = \Sigma_\eta^{-1} + \mathrm{diag}\!\left(\tfrac{D^{(n)}}{(\omega^{(n)}_t)^2}\right),\quad h = \Sigma_\eta^{-1}\bar\eta + \tfrac{1}{(\omega^{(n)}_t)^2}\sum_d x^{\text{Non-EV},(n)}_{d,t}.$$

$\Sigma_\eta^{-1}$ is computed once per iter via the Woodbury identity
($O(T r^2)$); the $T \times T$ Cholesky of $\Lambda$ is $\sim 10^5$ flops at
$T{=}96$ — negligible.

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

**Code (planned).** Per-home $\hat\eta^{(n)}$ and the conditional sampler will
live in a new `_fit_background_hier()` / `_sample_eta()` pair in
[`graphical_model.py`](../models/graphical_model.py). Not yet implemented.

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

**Code (planned).** New `_fit_eta_prior()` in
[`graphical_model.py`](../models/graphical_model.py).

### 2.3 $\omega^{(n)}_t$ — per-home Non-EV std-dev profile

**Distribution.** Independently across $t$ and $n$:

$$(\omega^{(n)}_t)^2 \stackrel{\text{iid}}{\sim} \mathrm{InvGamma}(a^\omega_t, b^\omega_t).$$

This is the conjugate prior for an unknown variance under Gaussian observations.

**Fit.** Per-home, per-time empirical variance against the fitted mean:

$$\widehat{(\omega^{(n)}_t)^2} = \tfrac{1}{D^{(n)}}\sum_{d=1}^{D^{(n)}} \left(x^{\text{Non-EV},(n)}_{d,t} - \hat\eta^{(n)}_t\right)^2.$$

**Inference.** Gibbs block 5 (see §4). Conditional on the augmented latent
$\{x^{\text{Non-EV},(n)}_{d,t}\}$ (§2.5) and current $\eta^{(n)}$,

$$(\omega^{(n)}_t)^2 \sim \mathrm{InvGamma}\!\left(a^\omega_t + \tfrac{D^{(n)}}{2},\ b^\omega_t + \tfrac{1}{2}\sum_d (x^{\text{Non-EV},(n)}_{d,t} - \eta^{(n)}_t)^2\right),$$

independently per $t$.

**Why per-home, per-$t$?** The rank-1 model used a *globally pooled*
$\sigma^{\text{Non-EV}}_t$ — every home shares the same noise profile. This is
clearly wrong: homes differ in HVAC cycling, appliance usage, baseline
variability. We let homes have their own scale; the hierarchical IG prior
shares strength across homes.

**Why diagonal across $t$ (independent IG per $t$)?** This is the
*HMM-compatibility constraint*: emission variance enters the FFBS forward
pass diagonally. A non-diagonal prior on $\omega$ across $t$ would *not*
break the HMM (we Gibbs-sample $\omega$ in its own block, so the HMM never
sees a non-diagonal $\omega$ distribution) — but it would break conjugacy.
Independent IG per $t$ gives closed-form sampling; coupling across $t$ (e.g.,
log-Normal hierarchy on $\log\omega$) would require Metropolis or HMC. Not
worth the complexity for diminishing returns.

**Alternatives considered.**
- **Log-Normal hierarchy.** More "natural" prior on a positive scale; loses
  conjugacy. Rejected on simplicity grounds.
- **Pooled $\omega_t$ (rank-1's choice).** Simpler but empirically restrictive.
- **Half-Cauchy on $\omega^{(n)}_t$** with a higher-level scale. Common in
  Bayesian hierarchical literature; loses conjugacy. Same trade-off as
  log-Normal.

**Code (planned).** New `_sample_omega()` in
[`graphical_model.py`](../models/graphical_model.py).

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

**Code (planned).** New `_fit_omega_prior()` in
[`graphical_model.py`](../models/graphical_model.py).

### 2.5 $x^{\text{EV},(n)}_{d,t}, x^{\text{Non-EV},(n)}_{d,t}$ — latent decomposition (data augmentation)

**Why this is new at inference.** In the deprecated model, the Non-EV noise
$\sigma^{\text{Non-EV}}_t$ is a *fixed* global parameter at inference time, so
the combined emission variance $\sigma^2_{k,t}$ is constant across Gibbs
iterations and the $\Theta, \alpha$ conditionals are conjugate without
needing to decompose $x = x^{\text{EV}} + x^{\text{Non-EV}}$. In the new model,
$\omega^{(n)}_t$ is *sampled* at inference time — and Inverse-Gamma is **not**
conjugate to a Gaussian likelihood whose variance is $(\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2$
(a *sum* involving the unknown). To restore conjugacy, we sample the latent
decomposition.

**Distribution (closed-form Gaussian).** Conditional on $z^{(n)}_{d,t}=k$,
$\Theta^{(n)}, \eta^{(n)}, \omega^{(n)}$, and the observed total
$x^{(n)}_{d,t}$:

$$x^{\text{EV},(n)}_{d,t} \mid \cdot \;\sim\; \mathcal{N}\!\left(\mu^{\text{EV}}_{d,t},\ V_{d,t}\right),\quad x^{\text{Non-EV},(n)}_{d,t} = x^{(n)}_{d,t} - x^{\text{EV},(n)}_{d,t},$$

with

$$\mu^{\text{EV}}_{d,t} = \Theta^{(n)}_k + \tfrac{(\sigma^{\text{EV}}_k)^2}{(\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2}\!\left(x^{(n)}_{d,t} - \Theta^{(n)}_k - \eta^{(n)}_t\right),\quad V_{d,t} = \tfrac{(\sigma^{\text{EV}}_k)^2(\omega^{(n)}_t)^2}{(\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2}.$$

i.i.d. across $(d,t)$. Standard Gaussian-sum decomposition (Kalman-style).

**Cost.** $D\cdot T$ Gaussian draws per home per Gibbs iter ($\approx 35{,}000$
draws — vectorizable, $\ll$ FFBS cost).

**Memory.** Two $(D, T)$ arrays per home (~0.5 MB total). Negligible.

**Side benefit.** Once the latent decomposition is sampled, the other blocks
*simplify*: $\Theta^{(n)}_k$ updates use $\{x^{\text{EV}}_{d,t}\}$ directly
with variance $(\sigma^{\text{EV}}_k)^2$ alone (no per-$t$ combined variance);
$\eta^{(n)}$ updates use $\{x^{\text{Non-EV}}_{d,t}\}$ directly with variance
$(\omega^{(n)}_t)^2$ alone. Every block becomes a clean conjugate update.

**Trade-off.** Data augmentation introduces autocorrelation between successive
Gibbs samples (the augmented latents and the variance share information).
Mixing can be slower than a collapsed sampler. Standard, well-understood;
mitigations include longer burn-in or thinned chains if it becomes an issue
empirically.

**Alternative considered: non-conjugate $\omega$ sampling.** Skip the
augmentation block; sample $\omega^{(n)}_t$ via slice sampler or
Metropolis-Hastings on the combined-emission likelihood. Avoids the extra
$DT$ draws but introduces tuning (step size) and is harder to vectorize
cleanly. Rejected for simplicity.

**Code (planned).** New `_sample_latent_decomp()` block in the Gibbs loop.

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
likelihoods in block 1, the latent decomposition in block 2 (new model only),
and the data terms in the remaining blocks.

**Code.** Total-power arrays are assembled per home by
[`_build_home_arrays()`](../models/graphical_model.py#L230); used throughout
[`infer_home()`](../models/graphical_model.py#L490).

---

## 4. Inference loop (cross-cutting)

Per-home Gibbs sampler ([`infer_home()`](../models/graphical_model.py#L490)),
applied only to homes with $\hat C^{(n)}=1$.

### 4.1 New model — five-block Gibbs

Each iteration executes the following blocks in order. **Not yet implemented.**

1. **FFBS for $z^{(n)}_{d,t}$** (§1.3). Uses combined-emission likelihood
   $\mathcal{N}(\Theta^{(n)}_k + \eta^{(n)}_t,\ (\sigma^{\text{EV}}_k)^2 + (\omega^{(n)}_t)^2)$.
   Factorizes across $t$ given the current parameters — diagonal emission
   variance is required and preserved.
2. **Latent decomposition** (§2.5). Sample $x^{\text{EV},(n)}_{d,t}$ from the
   Gaussian-sum conditional given observed $x^{(n)}_{d,t}$, $z^{(n)}_{d,t}$,
   $\Theta^{(n)}, \eta^{(n)}, \omega^{(n)}$. Set
   $x^{\text{Non-EV},(n)}_{d,t} = x^{(n)}_{d,t} - x^{\text{EV},(n)}_{d,t}$.
3. **$\Theta^{(n)}_k$** (§1.5, simplified). Conjugate Gaussian using
   $\{x^{\text{EV},(n)}_{d,t} : z^{(n)}_{d,t}{=}k\}$ with variance
   $(\sigma^{\text{EV}}_k)^2$ alone (no per-$t$ combined variance term).
4. **$\eta^{(n)}$** (§2.1). $T$-dim Gaussian conjugate update from
   $\{x^{\text{Non-EV},(n)}_{d,t}\}$ with PPCA prior $\mathcal{N}(\bar\eta, \Sigma_\eta)$
   and per-$t$ likelihood variance $(\omega^{(n)}_t)^2$. $T\times T$ Cholesky;
   Woodbury for $\Sigma_\eta^{-1}$.
5. **$\omega^{(n)}_t$** (§2.3). Per-$t$ Inverse-Gamma conjugate update from
   residuals $\{x^{\text{Non-EV},(n)}_{d,t} - \eta^{(n)}_t\}_d$.

**Initialization.** $\eta^{(n)} = \bar\eta$, $\omega^{(n)}_t = $ prior mode
$\sqrt{b^\omega_t / (a^\omega_t + 1)}$, $\Theta^{(n)}_k = \mu_{\Theta_k}$,
$z\equiv\texttt{off}$.

**Schedule.** Keep $S_{\text{burn}}=200$ burn-in + $S=500$ retained iters.
Re-tune empirically after Tier-1 evaluation; data augmentation may require
longer burn-in.

**Accumulation.** $z$-counts accumulated incrementally post-burn (unchanged).
Per-iter $\eta^{(n)}, \omega^{(n)}, \Theta^{(n)}$ samples retained
(cheap; $S \times T$ and $S\times K$ floats per home).

**Computational budget.** FFBS still dominates ($O(K^2 T D)$); blocks 2 and 5
are $O(DT)$; block 4 is $O(T^3 + Tr^2) \approx 10^6$ flops per iter. Per-home
per-iter cost increases by $\sim 2\times$ vs. the deprecated model; total
runtime still well under 10 minutes for 9 EV homes.

### 4.2 Deprecated model — three-block Gibbs

The current code implements this version:

- **Initialization.** $\alpha^{(n)} = \mu_\alpha$, $\Theta^{(n)}_k = \mu_{\Theta_k}$,
  $z\equiv\texttt{off}$.
- **Schedule.** $S_{\text{burn}}=200$ burn-in + $S=500$ retained iterations,
  blocks executed in order 1 ($z$ via FFBS) → 2 ($\Theta$) → 3 ($\alpha$).
- **Accumulation.** $z$-counts accumulated incrementally post-burn; per-iter
  $\alpha$ and $\Theta$ samples retained.

### 4.3 Cross-cutting

- **Log-likelihood tracking.** [`_compute_loglik()`](../models/graphical_model.py#L667)
  for EV homes; [`_compute_loglik_c0()`](../models/graphical_model.py#L681) for
  non-EV homes (closed-form, no Gibbs). Both will need to be re-derived for
  the new model.
- The driver [`infer_all()`](../models/graphical_model.py#L816) iterates over
  homes; [`c_prob_from_z_via_heuristic()`](../models/graphical_model.py#L877) and
  [`build_heuristic_homes()`](../models/graphical_model.py#L896) bridge the
  heuristic detector into the per-home inference pipeline. Unchanged across
  migration.
- **Memory per home.** Dominated by $z$-counts of shape $(D, T, K)$ — under
  1 MB. New model adds two $(D,T)$ arrays for augmented latents (~0.5 MB).

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

### 6.1 Shared (EV side and total)

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $C^{(n)}$ | per-home latent | observed | heuristic | [`first_diff_logistic.predict`](../models/first_diff_logistic.py#L94) |
| $p_C$ | global scalar | empirical mean | unused | [`fit()`](../models/graphical_model.py#L129) |
| $z^{(n)}_{d,t}$ | per-home latent | observed | FFBS (block 1) | [`_ffbs()`](../models/graphical_model.py#L702) |
| $\pi_z, P_z$ | global | smoothed counts | read-only | [`_fit_hmm()`](../models/graphical_model.py#L257) |
| $\Theta^{(n)}_k$ | per-home latent | observed | Gibbs (block 3, new; block 2, old) | [`_sample_theta_k()`](../models/graphical_model.py#L789) |
| $\mu_{\Theta_k}, \sigma_{\Theta_k}, \sigma^{\text{EV}}_k$ | global | EM | read-only | [`_fit_charging_em()`](../models/graphical_model.py#L366) |
| $x^{(n)}_{d,t}$ | observed | observed | observed | [`_build_home_arrays()`](../models/graphical_model.py#L230) |

### 6.2 Non-EV side — new model (planned)

| Symbol | Kind | Fit | Inference | Code (planned) |
|---|---|---|---|---|
| $\eta^{(n)}_t$ | per-home latent ($T$-vec) | empirical day-mean | Gibbs block 4 ($T$-dim Gaussian) | `_sample_eta()` |
| $\bar\eta_t, W, \psi$ | global ($\Sigma_\eta = WW^\top + \mathrm{diag}(\psi)$) | mean + PPCA-EM with bias-correction | read-only | `_fit_eta_prior()` |
| $\omega^{(n)}_t$ | per-home latent ($T$-vec) | empirical per-$t$ var | Gibbs block 5 (per-$t$ IG) | `_sample_omega()` |
| $a^\omega_t, b^\omega_t$ | global ($T$-vec each) | method-of-moments | read-only | `_fit_omega_prior()` |
| $x^{\text{EV}}, x^{\text{Non-EV}}$ | augmented latents | (latents at fit-time too, but observed) | Gibbs block 2 (Gaussian-sum decomp) | `_sample_latent_decomp()` |

### 6.3 Non-EV side — deprecated rank-1 (currently in code)

| Symbol | Kind | Fit | Inference | Code |
|---|---|---|---|---|
| $\alpha^{(n)}$ | per-home latent (scalar) | plug-in OLS | Gibbs block 3 | [`_sample_alpha()`](../models/graphical_model.py#L789) |
| $\mu_\alpha, \sigma_\alpha^2$ | global | mean / bias-corrected var | read-only | [`_fit_background()`](../models/graphical_model.py#L300) |
| $\rho_t$ | global ($T$-vec) | top right SVD vector | read-only | [`_fit_background()`](../models/graphical_model.py#L300) |
| $\sigma^{\text{Non-EV}}_t$ | global ($T$-vec) | individual-obs MSE | read-only | [`_fit_background()`](../models/graphical_model.py#L300) |
