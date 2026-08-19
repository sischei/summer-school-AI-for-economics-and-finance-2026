# Day 1 — Scheidegger: Notebooks

Companion notebooks for the three Day 1 sessions. Slides live in
[`../slides`](../slides); the papers they cite are in [`../readings`](../readings).

Everything runs on [nuvolos.cloud](https://nuvolos.cloud) with no local setup. Locally, the
notebooks need Python 3.10+, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tensorflow`
(2.x) and `torch`.

---

## Part I — Introduction to Deep Learning and Deep Equilibrium Nets (09:10 – 10:30)

Slides: [`01_Intro_to_DeepLearning.pdf`](../slides/01_Intro_to_DeepLearning.pdf)

| # | Notebook | Topic | Slides | Stack | Role |
|---|---|---|---|---|---|
| 01 | [`01_01_BasicML_intro.ipynb`](01_deep_learning_intro/01_01_BasicML_intro.ipynb) | Linear regression, classification, k-means, and loss functions | Part 1 | scikit-learn | core |
| 02 | [`01_02_GradientDescent_and_StochasticGradientDescent.ipynb`](01_deep_learning_intro/01_02_GradientDescent_and_StochasticGradientDescent.ipynb) | Gradient descent and SGD from scratch | Part 3 | numpy | core |
| 03 | [`01_03_Double_Descent.ipynb`](01_deep_learning_intro/01_03_Double_Descent.ipynb) | The double-descent phenomenon | Part 4 | numpy | core |
| 04 | [`01_04_Gentle_DNN.ipynb`](01_deep_learning_intro/01_04_Gentle_DNN.ipynb) | A first deep network: regression and classification | Part 2 | TensorFlow/Keras | core |
| 05 | [`01_05_PyTorch_intro.ipynb`](01_deep_learning_intro/01_05_PyTorch_intro.ipynb) | The same two tasks in PyTorch | Part 5 | PyTorch | core |
| 06 | [`01_06_Genz_Approximation_and_Loss_Functions.ipynb`](01_deep_learning_intro/01_06_Genz_Approximation_and_Loss_Functions.ipynb) | Genz test functions, loss kernels, and the curse of dimensionality | Parts 3 and 5 | TensorFlow/Keras | warm-up exercise |

`01_02` reads [`SGD_data.txt`](01_deep_learning_intro/SGD_data.txt) from its own directory.

---

## Part II — Deep Equilibrium Nets (11:00 – 12:30)

Slides: [`02_DEQN.pdf`](../slides/02_DEQN.pdf)

| # | Notebook | Topic | Slides | Role |
|---|---|---|---|---|
| 01 | [`02_01_Brock_Mirman_1972_DEQN.ipynb`](02_deep_equilibrium_nets/02_01_Brock_Mirman_1972_DEQN.ipynb) | Deterministic Brock–Mirman; hard vs. soft constraints; validation against $s^\star = \alpha\beta$ | Part III | core |
| 02 | [`02_02_Brock_Mirman_Uncertainty_DEQN.ipynb`](02_deep_equilibrium_nets/02_02_Brock_Mirman_Uncertainty_DEQN.ipynb) | Stochastic Brock–Mirman; simulation-based sampling; Gauss–Hermite quadrature | Part III | core |
| 03 | [`02_03_DEQN_Exercises_Blanks.ipynb`](02_deep_equilibrium_nets/02_03_DEQN_Exercises_Blanks.ipynb) | Guided exercises, including KKT via the Fischer–Burmeister residual | Part III | exercise |
| 04 | [`02_04_DEQN_Exercises_Solutions.ipynb`](02_deep_equilibrium_nets/02_04_DEQN_Exercises_Solutions.ipynb) | Solutions to notebook 03 | Part III | solution |
| 05 | [`02_05_StochasticBM_LossComparison.ipynb`](02_deep_equilibrium_nets/02_05_StochasticBM_LossComparison.ipynb) | Six loss kernels on the same model: MSE, MAE, Huber, pinball, CVaR, log-cosh | Part III | core |
| 06 | [`02_06_Grid_vs_Random_Search.ipynb`](02_deep_equilibrium_nets/02_06_Grid_vs_Random_Search.ipynb) | Where to place a fixed search budget: grid vs. random, and when each wins | Part V | core |
| 07 | [`02_07_NAS_RandomSearch_Hyperband.ipynb`](02_deep_equilibrium_nets/02_07_NAS_RandomSearch_Hyperband.ipynb) | Random search and successive halving from scratch, compared per epoch spent | Part V | core |
| 08 | [`02_08_Loss_Normalization.ipynb`](02_deep_equilibrium_nets/02_08_Loss_Normalization.ipynb) | Multi-component losses: non-dimensionalization, inverse-loss weighting, ReLoBRaLo | Part IV | core |

All Part II notebooks are TensorFlow/Keras. `02_07` caches its search records in
`nas_results/search_records.pkl` so that re-runs are instant; the cache records the
`RUN_MODE` it was written under and is ignored if you switch modes. Delete it to force a
fresh sweep.

---

## Part III — Deep Surrogates to Estimate Dynamic Models (13:30 – 15:00)

Slides: [`03_Deep_Surrogates.pdf`](../slides/03_Deep_Surrogates.pdf)

| # | Notebook | Topic | Slides | Runtime | Role |
|---|---|---|---|---|---|
| 01 | [`03_01_Surrogate_Primer.ipynb`](03_deep_surrogates/03_01_Surrogate_Primer.ipynb) | A Black–Scholes deep surrogate over $(S, K, T, \sigma, r)$, then implied-volatility inversion by root-finding *and* by gradient descent through the network | Part I | ~3.5 min | core |
| 02 | [`03_02_GP_and_BAL.ipynb`](03_deep_surrogates/03_02_GP_and_BAL.ipynb) | GP regression from scratch in numpy, checked against `scikit-learn`; then Bayesian active learning and a budget sweep against a uniform design | Part III | ~30 s | core |
| 03 | [`03_03_Structural_Estimation_BM.ipynb`](03_deep_surrogates/03_03_Structural_Estimation_BM.ipynb) | Scalar SMM: train the surrogate with the TFP persistence $\varrho$ as a pseudo-state, simulate under common random numbers, recover $\varrho$ | Part II | ~30 s | core |
| 04 | [`03_04_Structural_Estimation_BM_Joint.ipynb`](03_deep_surrogates/03_04_Structural_Estimation_BM_Joint.ipynb) | Joint $(\beta, \varrho)$ estimation, the identification ridge and how extra moments close it, the moment Jacobian's SVD, and a surrogate health check | Part II | ~60 s | core |

All four are **PyTorch + scikit-learn**, CPU-only, and self-contained: they read no files, and
they write none unless you flip `SAVE_FIGURES`. (The two SMM notebooks then write into
`../../slides/figures`.) Runtimes above are for `RUN_MODE = "smoke"` on a laptop CPU. Figures are emitted at
presentation font sizes (16 pt base): nothing inside a figure should end up smaller than the
slides' `\footnotesize`.

`03_01` and `03_02` carry a `RUN_MODE` switch (`"smoke"` / `"teaching"` / `"production"`) in
their first cell. They ship on `"smoke"`; raise it for publication-quality figures at
proportionally higher cost. `03_03` and `03_04` have fixed classroom budgets instead.

> **Three repairs were made to the ported notebooks; they are worth knowing about if you
> compare against the originals.**
>
> * **`RUN_MODE` did not control anything.** `03_01` hard-coded its training-set size,
>   epoch count and batch size at the `"teaching"` values, and `03_02` hard-coded its BAL
>   iteration count, so selecting `"smoke"` changed nothing but the label. Both now read the
>   dispatched budget, which is what cuts `03_01` from ~15 minutes to ~3.5.
> * **`03_02`'s active-learning loop could re-select points it already had.** The acquisition
>   took an `argmax` of posterior variance over the *whole* candidate grid, including points
>   already in the design. A duplicated row makes the kernel matrix singular, which drives the
>   marginal-likelihood optimiser to its bounds and collapses the GP. The loop now excludes
>   points already queried.
> * **The GP length scale was unbounded.** Combined with the above, the optimiser ran to
>   `length_scale = 1e-5` and the posterior rang back to the prior between every observation —
>   the exact pathology on the *"Getting the Length Scale Wrong"* slide. All GPs in the BAL
>   section now use `length_scale_bounds = (1e-2, 1e1)` on a domain of width 6, and a jitter
>   of `1e-6`. Before these two fixes `03_02` **failed its own closing assertion** and did not
>   run to completion.

> **A fourth repair, made during the pre-teaching review.** `03_04` shipped with a
> 200-step training budget. That left the policy **32% off** the closed-form steady-state
> anchor and, worse, **wrong-signed in $\beta$**: it reported
> $\partial(\text{mean savings})/\partial\beta = -0.0016$ where the closed form gives
> $+1.92$. The joint SMM criterion then showed an "identification ridge" in $\beta$ that is
> **not in the model** — condition number 305 — and the deck presented it as an econometric
> finding. It was the *lazy-learning* pathology from the morning's Part VI.
>
> Both SMM notebooks are now trained to convergence (`03_03`: 3,000 steps; `03_04`: 8,000
> steps, hidden width 128, anchor weight 0.3). The anchor error falls to 0.0%, and the picture
> that emerges is the textbook one: with the **two dynamic moments** the criterion has a real
> ridge (condition number **96**), and adding the **level moments** closes it (condition number
> **2.5**). `03_04` now prints both condition numbers and a surrogate-health table comparing the
> learned policy against $s^\star(\beta)$ at five values of $\beta$ — run that check before
> interpreting any criterion surface.

**One result worth flagging,** because it corrects a claim often made in passing: on the 1-D
target used here, Bayesian active learning does **not** beat a uniform design (MAE 0.0234 vs
0.0237 at $N = 25$; the two curves interleave throughout). That is the theory working. For a
*stationary* kernel the posterior variance is a function of the **design** alone, not of the
values observed, so pure uncertainty sampling on an interval essentially reproduces a
space-filling design. Active learning earns its keep where no space-filling design is
available or affordable — irregular domains, ARD kernels that need different resolutions in
different directions, and dimensions high enough that a grid is out of the question. That is
the carbon-tax application in Part V of the slides, not this notebook.

---

The three exercises previewed on the hands-on slide of `02_DEQN.pdf` map onto notebooks
`02_01` (deterministic benchmark), `02_02` (stochastic, with quadrature) and
`02_03`/`02_04` (the KKT extension).
