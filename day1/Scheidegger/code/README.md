# Day 1 — Scheidegger: Notebooks

Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tensorflow` (2.x) and
`torch`. Runs on [nuvolos.cloud](https://nuvolos.cloud) with no local setup.

---

## Part I — Introduction to Deep Learning and Deep Equilibrium Nets (09:10 – 10:30)

Slides: [`01_Intro_to_DeepLearning.pdf`](../slides/01_Intro_to_DeepLearning.pdf)

| # | Notebook | Topic | Stack |
|---|---|---|---|
| 01 | [`01_01_BasicML_intro.ipynb`](01_deep_learning_intro/01_01_BasicML_intro.ipynb) | Linear regression, classification, k-means, loss functions | scikit-learn |
| 02 | [`01_02_GradientDescent_and_StochasticGradientDescent.ipynb`](01_deep_learning_intro/01_02_GradientDescent_and_StochasticGradientDescent.ipynb) | Gradient descent and SGD from scratch | numpy |
| 03 | [`01_03_Double_Descent.ipynb`](01_deep_learning_intro/01_03_Double_Descent.ipynb) | The double-descent phenomenon | numpy |
| 04 | [`01_04_Gentle_DNN.ipynb`](01_deep_learning_intro/01_04_Gentle_DNN.ipynb) | A first deep network: regression and classification | TensorFlow/Keras |
| 05 | [`01_05_PyTorch_intro.ipynb`](01_deep_learning_intro/01_05_PyTorch_intro.ipynb) | The same two tasks in PyTorch | PyTorch |
| 06 | [`01_06_Genz_Approximation_and_Loss_Functions.ipynb`](01_deep_learning_intro/01_06_Genz_Approximation_and_Loss_Functions.ipynb) | Genz test functions, loss kernels, the curse of dimensionality | TensorFlow/Keras |

Data: [`SGD_data.txt`](01_deep_learning_intro/SGD_data.txt), read by `01_02`.

---

## Part II — Deep Equilibrium Nets (11:00 – 12:30)

Slides: [`02_DEQN.pdf`](../slides/02_DEQN.pdf) · TensorFlow/Keras throughout

| # | Notebook | Topic | Role |
|---|---|---|---|
| 01 | [`02_01_Brock_Mirman_1972_DEQN.ipynb`](02_deep_equilibrium_nets/02_01_Brock_Mirman_1972_DEQN.ipynb) | Deterministic Brock–Mirman; hard vs. soft constraints; validation against $s^\star = \alpha\beta$ | core |
| 02 | [`02_02_Brock_Mirman_Uncertainty_DEQN.ipynb`](02_deep_equilibrium_nets/02_02_Brock_Mirman_Uncertainty_DEQN.ipynb) | Stochastic Brock–Mirman; simulation-based sampling; Gauss–Hermite quadrature | core |
| 03 | [`02_03_DEQN_Exercises_Blanks.ipynb`](02_deep_equilibrium_nets/02_03_DEQN_Exercises_Blanks.ipynb) | Guided exercises, including KKT via the Fischer–Burmeister residual | exercise |
| 04 | [`02_04_DEQN_Exercises_Solutions.ipynb`](02_deep_equilibrium_nets/02_04_DEQN_Exercises_Solutions.ipynb) | Solutions to notebook 03 | solution |
| 05 | [`02_05_StochasticBM_LossComparison.ipynb`](02_deep_equilibrium_nets/02_05_StochasticBM_LossComparison.ipynb) | Six loss kernels: MSE, MAE, Huber, pinball, CVaR, log-cosh | core |
| 06 | [`02_06_Grid_vs_Random_Search.ipynb`](02_deep_equilibrium_nets/02_06_Grid_vs_Random_Search.ipynb) | Grid vs. random search for a fixed budget | core |
| 07 | [`02_07_NAS_RandomSearch_Hyperband.ipynb`](02_deep_equilibrium_nets/02_07_NAS_RandomSearch_Hyperband.ipynb) | Random search and successive halving, compared per epoch spent | core |
| 08 | [`02_08_Loss_Normalization.ipynb`](02_deep_equilibrium_nets/02_08_Loss_Normalization.ipynb) | Non-dimensionalization, inverse-loss weighting, ReLoBRaLo | core |

Cache: `02_07` writes `nas_results/search_records.pkl`; delete it to force a fresh sweep.

---

## Part III — Deep Surrogates to Estimate Dynamic Models (13:30 – 15:00)

Slides: [`03_Deep_Surrogates.pdf`](../slides/03_Deep_Surrogates.pdf) · PyTorch and
scikit-learn, CPU-only

| # | Notebook | Topic | Runtime |
|---|---|---|---|
| 01 | [`03_01_Surrogate_Primer.ipynb`](03_deep_surrogates/03_01_Surrogate_Primer.ipynb) | Black–Scholes deep surrogate over $(S, K, T, \sigma, r)$; implied-volatility inversion by root-finding and by gradient descent | ~3.5 min |
| 02 | [`03_02_GP_and_BAL.ipynb`](03_deep_surrogates/03_02_GP_and_BAL.ipynb) | GP regression from scratch and via scikit-learn; Bayesian active learning against a uniform design | ~30 s |
| 03 | [`03_03_Structural_Estimation_BM.ipynb`](03_deep_surrogates/03_03_Structural_Estimation_BM.ipynb) | Scalar SMM: $\varrho$ as a pseudo-state, common random numbers, grid estimate | ~30 s |
| 04 | [`03_04_Structural_Estimation_BM_Joint.ipynb`](03_deep_surrogates/03_04_Structural_Estimation_BM_Joint.ipynb) | Joint $(\beta, \varrho)$ SMM; criterion surfaces, moment Jacobian SVD, surrogate health check | ~60 s |

`03_01` and `03_02` carry a `RUN_MODE` switch (`"smoke"` / `"teaching"` / `"production"`) in
their first cell and ship on `"smoke"`; runtimes above are for that setting. Figures are
written only when `SAVE_FIGURES` is set, into [`../slides/figures`](../slides/figures).
