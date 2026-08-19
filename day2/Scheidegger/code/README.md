# Day 2 — Scheidegger: Notebooks

Physics-Informed Neural Networks for solving Partial Differential Equations (09:00 – 10:30).

Slides: [`04_PINNs.pdf`](../slides/04_PINNs.pdf) · Readings: [`../readings`](../readings)

Python 3.10+ with `torch`, `numpy`, `matplotlib` and `scipy`. All seven are PyTorch,
CPU-only, and perform no file I/O. Runs on [nuvolos.cloud](https://nuvolos.cloud) with no
local setup.

| # | Notebook | Topic | Slides | Role |
|---|---|---|---|---|
| 01 | [`01_ODE_PINN_ZeroBCs.ipynb`](01_ODE_PINN_ZeroBCs.ipynb) | The smallest complete PINN: $y''=-1$ on $(0,1)$ with zero Dirichlet BCs, against the closed form | Part II | in-class walkthrough |
| 02 | [`02_ODE_PINN_SoftVsHardBCs.ipynb`](02_ODE_PINN_SoftVsHardBCs.ipynb) | The same ODE with non-zero BCs: soft penalty vs. hard trial solution | Part III | in-class walkthrough |
| 03 | [`03_PDE_PINN_Poisson2D.ipynb`](03_PDE_PINN_Poisson2D.ipynb) | 2D Poisson on the unit square; hard BCs by transfinite interpolation | Part III | self-study |
| 04 | [`04_Cake_Eating_HJB_PINN.ipynb`](04_Cake_Eating_HJB_PINN.ipynb) | The cake-eating HJB equation as a scaled hard-BC trial solution; Adam → L-BFGS | Part IV | in-class walkthrough |
| 05 | [`05_Black_Scholes_PINN.ipynb`](05_Black_Scholes_PINN.ipynb) | European call pricing; the Greeks by automatic differentiation | Part V | self-study |
| 06 | [`06_PINN_Exercise.ipynb`](06_PINN_Exercise.ipynb) | Build a PINN from scratch for $u''+u=0$ on $[0,\pi]$ | Part VI | exercise (fill-in-the-blank; solutions in the same file) |
| 07 | [`07_PE_Discrete_HJB_PINN.ipynb`](07_PE_Discrete_HJB_PINN.ipynb) | Partial-equilibrium HJB with a 2-state income chain, by upwind finite differences and by a PINN | Part IV | self-study |
