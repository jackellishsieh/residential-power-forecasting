# Fit: Parameter Estimation

All notation follows `graphical_model.tex`. The training set fully observes
$C^{(n)}$, $z^{(n)}_{d,t}$, $x^{\textrm{EV},(n)}_{d,t}$, $x^{\textrm{Non-EV},(n)}_{d,t}$,
and $x^{(n)}_{d,t}$. All global parameters are estimated in closed form (or
short EM); none are sampled at fit time.

Let $\mathcal{N}^+ = \{n : C^{(n)}=1\}$, with $|\mathcal{N}^+| = N_{\textrm{EV}}$.
Per-home day count $D^{(n)}$ is **not constant** — it ranges from ~180 to ~360.
Estimators below handle this throughout.

---

## Step 1 — EV prevalence $p_C$

$$p_C = \frac{1}{N}\sum_n C^{(n)}$$

Trivial empirical mean.

---

## Step 2 — HMM parameters $\pi_z$, $P_z$ (EV homes only)

Estimated only on $\mathcal{N}^+$ since for non-EV homes the chain is fixed
(deterministic "off"). Each day is treated as an **independent chain** that
resets at $t{=}0$, so transitions are counted strictly within days
($t{-}1 \to t$ for $t \in \{1, \ldots, T{-}1\}$, never across day boundaries).

**Initial distribution.** Empirical frequency of day-start states:

$$\pi_z[k] = \frac{\sum_{n \in \mathcal{N}^+}\sum_{d=1}^{D^{(n)}} \mathbf{1}[z^{(n)}_{d,0} = k]}{\sum_{n \in \mathcal{N}^+} D^{(n)}}$$

**Transition matrix.** Add a small Laplace constant $\lambda = 10^{-3}$ to
guard against zero-probability transitions (with only $N_{\textrm{EV}}=9$
EV homes, rare transitions can go unobserved):

$$P_z[k, k'] = \frac{\lambda + \sum_{n \in \mathcal{N}^+}\sum_d \sum_{t=1}^{T-1} \mathbf{1}[z^{(n)}_{d,t-1}=k,\; z^{(n)}_{d,t}=k']}{K\lambda + \sum_{n \in \mathcal{N}^+}\sum_d \sum_{t=1}^{T-1} \mathbf{1}[z^{(n)}_{d,t-1}=k]}$$

**Why MLE here?** Multinomial counts → empirical frequencies are the MLE;
no random-effects structure to worry about. Smoothing is the only
practical safeguard against unobserved cells.

---

## Step 3 — Background block $\rho$, $\mu_\alpha$, $\sigma_\alpha$, $\sigma^{\textrm{Non-EV}}_t$

The non-EV submodel is a **rank-1 hierarchical factor model with per-time
heteroscedastic noise**:

$$\alpha^{(n)} \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2), \quad
x^{\textrm{Non-EV},(n)}_{d,t} \mid \alpha^{(n)} \sim \mathcal{N}\!\left(\alpha^{(n)} \rho_t,\; (\sigma^{\textrm{Non-EV}}_t)^2\right)$$

Per-home data is abundant (each home has $D^{(n)} \cdot T$ ≈ 17K–35K observations),
so a plug-in $\hat\alpha^{(n)}$ is essentially the MLE — formal EM would
move estimates by less than a fraction of a percent. We use a **closed-form
two-stage estimator with the bias correction that matters** (the only one
that's not negligible at this data scale: the variance estimator for
$\sigma^{\textrm{Non-EV}}_t$).

### 3a. Per-home day-mean profile

$$\hat{\beta}^{(n)}_t = \frac{1}{D^{(n)}} \sum_{d=1}^{D^{(n)}} x^{\textrm{Non-EV},(n)}_{d,t}$$

Stack as $B \in \mathbb{R}^{N \times T}$.

### 3b. Background profile $\rho$ via SVD

$B = U\Sigma V^\top$ (economy SVD); take $\rho = V_{:,1}$ (right singular
vector with largest singular value, unit norm by construction).

**Sign convention.** SVD gives $\rho$ up to sign; flip so that
$\mathrm{median}_n(\hat\alpha^{(n)}) > 0$, since background load is
non-negative and $\alpha^{(n)}$ represents scale.

**Why SVD over EM?** With $D^{(n)} \approx 200$–360 averaging out per-timestep
noise in $\hat\beta$, the heteroscedasticity in $B$'s columns is at the
level of $(\sigma^{\textrm{Non-EV}}_t)^2 / D^{(n)}$ — small relative to
the $\alpha^{(n)} \rho_t$ signal. Plain SVD vs heteroscedastically-weighted
SVD differ in the third decimal. EM would arrive at the same answer and
require iteration.

### 3c. Per-home scale (plug-in)

$$\hat\alpha^{(n)} = \hat\beta^{(n)} \cdot \rho = \sum_t \hat\beta^{(n)}_t \rho_t$$

This is the OLS projection of the home's day-mean profile onto $\rho$.

$$\mu_\alpha = \frac{1}{N} \sum_n \hat\alpha^{(n)}$$

### 3d. Per-time noise $\sigma^{\textrm{Non-EV}}_t$ — **use individual obs, not day-means**

$$(\sigma^{\textrm{Non-EV}}_t)^2 = \frac{1}{\sum_n D^{(n)}} \sum_n \sum_{d=1}^{D^{(n)}} \left(x^{\textrm{Non-EV},(n)}_{d,t} - \hat\alpha^{(n)} \rho_t\right)^2$$

**Why not residuals of $\hat\beta$?** Computing residuals from
day-means estimates $\mathrm{Var}(\hat\beta^{(n)}_t - \alpha^{(n)} \rho_t)$,
which is $(\sigma^{\textrm{Non-EV}}_t)^2 / D^{(n)}$ — off by a factor of $D \approx 250$
(or $\sqrt{D} \approx 16$ in std). The FFBS emission would be wildly overconfident.
Residuals from individual observations is the correct estimator and uses all
$\sum_n D^{(n)} \cdot T$ ≈ 1.5M data points.

### 3e. Cross-home scale variance $\sigma_\alpha$ — bias-corrected

$$\sigma_\alpha^2 = \max\!\left(0,\; \mathrm{Var}_n(\hat\alpha^{(n)}) - \frac{1}{N}\sum_n \frac{1}{D^{(n)}} \sum_t \rho_t^2 (\sigma^{\textrm{Non-EV}}_t)^2 \right)$$

The subtracted term is the noise variance of the plug-in $\hat\alpha^{(n)}$
(noise in $\hat\beta^{(n)}$ projected onto $\rho$, scaled by $1/D^{(n)}$).
**This correction is small** ($O(1/D^{(n)})$) but applied for symmetry
with Step 4 and to handle homes with smaller $D^{(n)}$.

---

## Step 4 — EV charging magnitude $\mu_{\Theta_k}$, $\sigma_{\Theta_k}$, $\sigma^{\textrm{EV}}_k$ (EM)

Two states need fitting: $k \in \{\mathtt{low}, \mathtt{high}\}$. The
$\mathtt{off}$ state is fixed: $\Theta^{(n)}_{\mathtt{off}} = 0$,
$\sigma_{\Theta_{\mathtt{off}}} = 0$, $\sigma^{\textrm{EV}}_{\mathtt{off}} = 10^{-3}$
(small floor for FFBS numerical stability).

This is the **one-way Gaussian random-effects model**:

$$\Theta^{(n)}_k \sim \mathcal{N}(\mu_{\Theta_k}, \sigma_{\Theta_k}^2),\quad
x^{\textrm{EV},(n)}_{d,t} \mid \Theta^{(n)}_k, z^{(n)}_{d,t}{=}k \sim \mathcal{N}\!\left(\Theta^{(n)}_k,\; (\sigma^{\textrm{EV}}_k)^2\right)$$

**Why EM and not closed-form ANOVA?** Per-home observation count $n^{(n)}_k$
(timesteps in state $k$ for home $n$) varies massively across homes — some
EV homes charge daily, others rarely. Under heavily unbalanced groups,
ANOVA $\neq$ MLE/REML; EM gives the MLE optimally weighting unbalanced groups
and adds only ~10 lines. Each iteration is closed-form; convergence in
$\lesssim 50$ iterations.

Define sufficient statistics per home (state $k$):

$$n^{(n)}_k = |\mathcal{T}^{(n)}_k|,\;\; S_y^{(n)} = \sum_{(d,t)\in\mathcal{T}^{(n)}_k} x^{\textrm{EV},(n)}_{d,t},\;\; SS_y^{(n)} = \sum_{(d,t)\in\mathcal{T}^{(n)}_k} (x^{\textrm{EV},(n)}_{d,t})^2$$

### 4a. Initialization (ANOVA)

$$\hat\theta^{(n)}_k = S_y^{(n)} / n^{(n)}_k$$

$$(\sigma^{\textrm{EV}}_k)^2_{(0)} = \frac{\sum_n SS_y^{(n)} - \sum_n n^{(n)}_k (\hat\theta^{(n)}_k)^2}{N_k - N_{\textrm{EV}}}, \quad N_k = \sum_n n^{(n)}_k$$

$$\sigma_{\Theta_k}^2{}_{(0)} = \max\!\left(0,\; \mathrm{Var}_n(\hat\theta^{(n)}_k) - (\sigma^{\textrm{EV}}_k)^2_{(0)} \cdot \tfrac{1}{N_{\textrm{EV}}}\sum_n \tfrac{1}{n^{(n)}_k}\right)$$

$$\mu_{\Theta_k\,(0)} = \frac{1}{N_{\textrm{EV}}} \sum_n \hat\theta^{(n)}_k$$

### 4b. EM iterations (until $|\Delta \log L| < 10^{-6}$ or 100 iters)

**E-step** (posterior of latent $\Theta^{(n)}_k$ under current params):

$$\mathrm{prec}^{(n)} = \frac{1}{\sigma_{\Theta_k}^2} + \frac{n^{(n)}_k}{(\sigma^{\textrm{EV}}_k)^2}$$

$$\mathbb{E}[\Theta^{(n)}_k] = \frac{\mu_{\Theta_k}/\sigma_{\Theta_k}^2 \;+\; S_y^{(n)} / (\sigma^{\textrm{EV}}_k)^2}{\mathrm{prec}^{(n)}}$$

$$\mathrm{Var}[\Theta^{(n)}_k] = 1/\mathrm{prec}^{(n)},\qquad \mathbb{E}[(\Theta^{(n)}_k)^2] = \mathrm{Var}[\Theta^{(n)}_k] + \mathbb{E}[\Theta^{(n)}_k]^2$$

If $\sigma_{\Theta_k}^2 = 0$, floor it to $10^{-6}$ to avoid division by zero.

**M-step** (closed form):

$$\mu_{\Theta_k} \leftarrow \frac{1}{N_{\textrm{EV}}} \sum_n \mathbb{E}[\Theta^{(n)}_k]$$

$$\sigma_{\Theta_k}^2 \leftarrow \frac{1}{N_{\textrm{EV}}} \sum_n \left(\mathrm{Var}[\Theta^{(n)}_k] + (\mathbb{E}[\Theta^{(n)}_k] - \mu_{\Theta_k})^2\right)$$

$$(\sigma^{\textrm{EV}}_k)^2 \leftarrow \frac{1}{N_k} \sum_n \left(SS_y^{(n)} - 2 S_y^{(n)} \mathbb{E}[\Theta^{(n)}_k] + n^{(n)}_k \mathbb{E}[(\Theta^{(n)}_k)^2]\right)$$

**Marginal log-likelihood** (monitor convergence; should be monotone non-decreasing):

$$\log L = -\tfrac{1}{2} \sum_n \left[ (n^{(n)}_k {-} 1) \log (\sigma^{\textrm{EV}}_k)^2 + \log\!\left((\sigma^{\textrm{EV}}_k)^2 + n^{(n)}_k \sigma_{\Theta_k}^2\right) + \frac{SS_y^{(n)}{-}n^{(n)}_k(\hat\theta^{(n)}_k)^2}{(\sigma^{\textrm{EV}}_k)^2} + \frac{n^{(n)}_k (\hat\theta^{(n)}_k - \mu_{\Theta_k})^2}{(\sigma^{\textrm{EV}}_k)^2 + n^{(n)}_k \sigma_{\Theta_k}^2} + n^{(n)}_k \log(2\pi)\right]$$

---

## Output

A `ModelParams` object containing all 11 parameter blocks:
$p_C$, $\pi_z$, $P_z$, $\rho$, $\mu_\alpha$, $\sigma_\alpha^2$,
$(\sigma^{\textrm{Non-EV}}_t)^2$, $\mu_{\Theta_k}$, $\sigma_{\Theta_k}^2$,
$(\sigma^{\textrm{EV}}_k)^2$.

Training is non-iterative apart from a short EM loop in Step 4 — total
fit time on the train set should be well under 10 seconds.
