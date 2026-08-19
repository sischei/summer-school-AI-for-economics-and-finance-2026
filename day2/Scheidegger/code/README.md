# Day 2 — Scheidegger: Notebooks

Companion notebooks for the Day 2 morning session, *Physics-Informed Neural Networks for
solving Partial Differential Equations* (09:00 – 10:30). Slides live in
[`../slides`](../slides); the paper they build on is in [`../readings`](../readings).

Everything runs on [nuvolos.cloud](https://nuvolos.cloud) with no local setup. Locally, the
notebooks need Python 3.10+, `torch`, `numpy`, `matplotlib` and `scipy`. All seven are
**PyTorch**, CPU-only, and self-contained — they read and write no files, so they can be run
from any directory.

---

## The notebooks

Slides: [`03_PINNs.pdf`](../slides/03_PINNs.pdf)

| # | Notebook | Topic | Slides | Role |
|---|---|---|---|---|
| 01 | [`01_ODE_PINN_ZeroBCs.ipynb`](01_ODE_PINN_ZeroBCs.ipynb) | The smallest complete PINN: $y''=-1$ on $(0,1)$ with zero Dirichlet BCs, against the closed form | Part II | in-class walkthrough |
| 02 | [`02_ODE_PINN_SoftVsHardBCs.ipynb`](02_ODE_PINN_SoftVsHardBCs.ipynb) | The same ODE with non-zero BCs: soft penalty vs. hard trial solution | Part III | in-class walkthrough |
| 03 | [`03_PDE_PINN_Poisson2D.ipynb`](03_PDE_PINN_Poisson2D.ipynb) | 2D Poisson on the unit square; hard BCs by transfinite interpolation | Part III | self-study |
| 04 | [`04_Cake_Eating_HJB_PINN.ipynb`](04_Cake_Eating_HJB_PINN.ipynb) | The cake-eating HJB equation as a scaled hard-BC trial solution; Adam → L-BFGS | Part IV | in-class walkthrough |
| 05 | [`05_Black_Scholes_PINN.ipynb`](05_Black_Scholes_PINN.ipynb) | European call pricing; the Greeks for free by automatic differentiation | Part V | self-study |
| 06 | [`06_PINN_Exercise.ipynb`](06_PINN_Exercise.ipynb) | Build a PINN from scratch for $u''+u=0$ on $[0,\pi]$ | Part VI | exercise |
| 07 | [`07_PE_Discrete_HJB_PINN.ipynb`](07_PE_Discrete_HJB_PINN.ipynb) | Partial-equilibrium HJB with a 2-state income chain, solved by upwind finite differences *and* by a PINN | Part IV (beyond) | self-study extension |

**Suggested order.** Start with `01`, `02` and `04` — these are the three we work through
together. Then `03` and `05` at your own pace, `06` when you want to write one yourself, and
`07` if you want to push past the lecture.

> **Note on `06`.** This is a fill-in-the-blank exercise: several cells contain `TODO`
> markers and a bare `pass`, so the notebook **will not run top-to-bottom unmodified**.
> A worked solution follows each task in the same file — try the blank first, then compare.

---

## Where the models come from

`01`–`03` are pure method: an ODE and a PDE whose exact solutions are known, so every design
choice can be scored against the truth rather than argued about. `04` and `05` are the two
economic applications — a continuous-time consumption-savings problem and an option-pricing
PDE — both of which also have closed forms.

`07` is the one problem here without a closed form, which is why it carries its own
benchmark: the same HJB is solved by an implicit upwind finite-difference scheme and by a
PINN, and the two are compared directly. On a 1D problem finite differences win outright.
Seeing exactly where and by how much is the point — it sets up the case for mesh-free
methods in the settings where grids stop being affordable, which is where the 11:00 – 12:30
session picks up.

The residual-based loss these notebooks minimise is the same idea as the Deep Equilibrium
Nets of Day 1 ([`../../day1/Scheidegger`](../../day1/Scheidegger)): the model's own equations
are the loss, and no labelled data is involved. What changes here is that the equations are
differential, so the derivatives come from `torch.autograd.grad` rather than from a
next-period expectation.
