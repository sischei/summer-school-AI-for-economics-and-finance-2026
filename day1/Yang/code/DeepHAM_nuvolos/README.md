# DeepHAM on Nuvolos

Code for the DeepHAM part of **Day 1, 15:30 – 17:00** (*Solving Heterogeneous Agent Models
with DeepHAM and Structural Reinforcement Learning*) and the **Day 2, 15:30 – 17:00**
practical session.

DeepHAM solves heterogeneous-agent models with aggregate shocks by (i) representing the
wealth distribution with a small number of **generalized moments** — basis functions learned
jointly with the solution rather than chosen by hand — and (ii) improving the policy by
differentiating through a simulation of the economy. The notebooks below build that up on
the Krusell–Smith (1998) model.

Paper: Han, Yang & E (2026), *DeepHAM: A global solution method for heterogeneous agent
models with aggregate shocks*, **Quantitative Economics** 17(2), 297–341.
Reference implementation: <https://github.com/frankhan91/DeepHAM>.

---

## The notebooks

All five live in [`src/`](src), because they import the DeepHAM modules (`param.py`,
`dataset.py`, `value.py`, `policy.py`) and read `../data`. Open them from there; the first
code cell locates the directory itself, so they also run from a local clone or from Colab.

| # | Notebook | Topic | Role | Runtime (`smoke`) |
|---|---|---|---|---|
| 01 | [`01_DeepHAM_KS_FixedMoment.ipynb`](src/01_DeepHAM_KS_FixedMoment.ipynb) | Krusell–Smith solved with the cross-sectional **mean** as the only distribution statistic (`n_fm=1, n_gm=0`) | in-class walkthrough | ~2 min |
| 02 | [`02_DeepHAM_KS_GeneralizedMoment.ipynb`](src/02_DeepHAM_KS_GeneralizedMoment.ipynb) | The same run with a **learned** generalized moment (`n_fm=0, n_gm=1`) — one config change | in-class walkthrough | ~2 min |
| 03 | [`03_DeepHAM_Policy_Exercise.ipynb`](src/03_DeepHAM_Policy_Exercise.ipynb) | Write the policy objective yourself: prices, budget constraint, `stop_gradient`, the unrolled utility sum | exercise (5 blanks) | ~2.5 min once filled |
| 04 | [`04_DeepHAM_Policy_Solutions.ipynb`](src/04_DeepHAM_Policy_Solutions.ipynb) | Solutions to 03, with commentary on the three details that matter | solution | ~2.5 min |
| 05 | [`05_DeepHAM_Visualize.ipynb`](src/05_DeepHAM_Visualize.ipynb) | What the generalized moment learned; reproduces Figures 3 and 7 of the paper | in-class walkthrough | ~20 s |

Notebook 05 needs no training: it loads the solved model shipped in `data/simul_results`.
Start there if you want the punchline before the machinery.

## Run modes

Every notebook opens with

```python
RUN_MODE = "smoke"   # one of: "smoke", "teaching", "production"
```

and a cell that maps it onto the handful of numbers that drive wall-clock time.

| | `smoke` | `teaching` | `production` |
|---|---|---|---|
| policy gradient steps | 100 | 1,500 | 10,000 |
| unroll horizon | 60 | 150 | 150 |
| simulated paths | 64 | 192 | 384 |
| value-net epochs | 10 | 60 | 200 |
| measured runtime (A100) | 2.2 min | 15.6 min | ~90 min |
| mean capital reached | ~11 | ~32 | ~39 (the KS level) |

`smoke` exercises every code path — dataset construction, value fitting, the policy loop,
the periodic value refresh, saving — but is far too short to converge: it lands around
$K \approx 11$ against the Krusell–Smith level of $K \approx 39$. `teaching` reaches about
32. `production` is the
setting behind the published results (the reference run in `data/simul_results` took
5,278 s).

Runs are written to `data/simul_results/KS/game_nn_n50_<exp>_<RUN_MODE>`, so a quick smoke
run can never overwrite a long one, nor the reference solutions below.

## What ships in `data/`

| | |
|---|---|
| `KS_policy_value_NS.mat` | the Krusell–Smith benchmark policy, as b-splines — the comparison target throughout |
| `simul_results/KS/game_nn_n50_1fm1` | solved model, 1 fixed moment (notebook 01) |
| `simul_results/KS/game_nn_n50_1gm3` | solved model, 1 generalized moment — the run behind Figure 3 of the paper (notebook 05) |

The `matlab/` folder holds the scripts that generate the benchmark `.mat` file; you do not
need to run them.

`src/` also contains the JFV and Dávila model code from the full DeepHAM repository
(`train_JFV.py`, `simulation_Davila.py`, …). Those are here so you can read them, but the
`.mat` inputs they need are not shipped — get them from the
[reference repository](https://github.com/frankhan91/DeepHAM) if you want to run them.

## Requirements

TensorFlow 2.x, NumPy, SciPy, matplotlib, tqdm. No GPU is required: at `smoke` and
`teaching` budgets the model is small enough that CPU is fine, and most of the wall-clock
goes into the NumPy simulation rather than the networks.

Verified on TensorFlow 2.20 / Keras 3.13 (Python 3.13). Note that `value.py` and `policy.py`
deliberately leave `prepare_state`, `value_fn` and `policy_fn` **undecorated**: nesting
`@tf.function` methods breaks from TF 2.20 onward, where `self` inside a traced method
becomes a `TfMethodTarget` that cannot resolve another `tf.function` attribute. They are
only ever called from inside `loss`/`train_step`, which are traced, so nothing is lost.
