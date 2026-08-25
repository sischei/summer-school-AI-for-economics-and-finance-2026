# DeepSAM on Nuvolos

Section 3 of **Deep Learning for Search and Matching Models**, prepared for **Day 2,
11:00 – 12:30** (*Deep Learning for Continuous-Time Models*) and the **Day 2, 15:30 – 17:00**
practical session. Slides:
[`../../slides/HACT_DeepSAM_Lecture_Slides.pdf`](../slides/HACT_DeepSAM_Lecture_Slides.pdf).

The model is a labour search-and-matching economy with **two-sided heterogeneity** (worker
types $x$, firm types $y$), **aggregate shocks**, and **distributional feedback**. Its state
is an aggregate shock $z$ together with the entire cross-sectional distribution $g$ of
existing matches — 55 dimensions here. DeepSAM learns the match surplus $S(x, y, z, g)$ as a
neural network taking $g$ directly as an input, trained on the residual of the master
equation at states drawn from simulating the model itself.

---

## The notebooks

| # | Notebook | What it does | Role | Runtime (`smoke`) |
|---|---|---|---|---|
| 01 | [`01_DeepSAM_Model_and_Solution.ipynb`](notebooks/01_DeepSAM_Model_and_Solution.ipynb) | The economics, the master equation, the deterministic steady states, and **how well the trained network satisfies the equation** across type space | in-class walkthrough | ~35 s |
| 02 | [`02_DeepSAM_COVID_Experiment.ipynb`](notebooks/02_DeepSAM_COVID_Experiment.ipynb) | The three results of Section 3: the COVID calibration, the recovery with and without re-sorting, and the mechanism behind the gap | in-class walkthrough | ~60 s |
| 03 | [`03_DeepSAM_Training.ipynb`](notebooks/03_DeepSAM_Training.ipynb) | What training the surplus network involves, and how far a short budget actually gets | practical session | ~60 s |

Notebooks 01 and 02 never train anything: they load the converged checkpoint shipped in
`checkpoints/`. That is the whole point of the split — the *economics* is cheap once the
network exists, and notebook 03 is where the cost of producing it is made explicit.

## Why this runs in minutes

The full Section 3 training run is a homotopy initialisation, a long main training phase
run to a loss threshold, and then **8 further rounds of 100,000 gradient steps** with the
simulated dataset rebuilt between rounds — several hours on an A100. Almost none of that is
needed to *understand* the method or to reproduce the figures:

* the trained checkpoint is loaded rather than re-derived;
* every remaining cost is a simulation cost, and each one is on a `RUN_MODE` dial;
* notebook 03 trains for 500 steps rather than 800,000, and reports the resulting gap
  (about 40× short of the checkpoint) so the cost of full convergence stays visible.

| | `smoke` | `teaching` | `production` |
|---|---|---|---|
| ergodic-pool paths × horizon | 32 × 500 | 64 × 1000 | 256 × 5000 |
| pre-COVID ergodic paths | 50 | 100 | 200 |
| recovery paths averaged | 20 | 60 | 200 |
| training steps (notebook 03) | 500 | 5,000 | 20,000 |

`production` matches the replication settings; the whole of notebook 02 still finishes in
around six minutes there, because the expensive part was always the training, not the
figures.

## What ships here

```
config/config.yaml      calibration, network size, training settings
src/train_nn.py         the method: steady states, networks, master-equation residual,
                        simulation of g, and the training loops
src/env.py              the economic environment: grids, production, matching, shocks
src/plotting.py         residual maps and the Section 3 figures
src/calibration_plot.py the COVID calibration figure
src/covid_shock_plot.py the recovery simulations
checkpoints/            the converged surplus networks (baseline and symmetric)
```

`outputs/` is created on demand. The `.npy` files that `solve_steady_state()` writes land in
the project root.

## Requirements

PyTorch (CUDA optional but much faster), NumPy, SciPy, matplotlib, `omegaconf`. Verified on
PyTorch 2.11 + CUDA 12.8 on an A100.

Two implementation notes:

* **Configuration.** `config/config.yaml` is loaded with `OmegaConf.load` and passed to
  `Train_NN` as keyword arguments — same file, same values, no extra dependency.
* **Logging.** `train_with_ergodic_loaders` takes optional `train_losses`/`eval_losses`
  lists that default to `None`, so each call starts with fresh logs; pass your own lists to
  accumulate across calls.
