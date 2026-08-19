# Day 1 — Scheidegger: Notebooks

Companion notebooks for the two Day 1 morning sessions. Slides live in
[`../slides`](../slides); the papers they cite are in [`../readings`](../readings).

Everything runs on [nuvolos.cloud](https://nuvolos.cloud) with no local setup. Locally, the
notebooks need Python 3.10+, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow` (2.x) and
`torch`.

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

The three exercises previewed on the hands-on slide of `02_DEQN.pdf` map onto notebooks
`02_01` (deterministic benchmark), `02_02` (stochastic, with quadrature) and
`02_03`/`02_04` (the KKT extension).
