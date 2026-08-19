"""
Train a small Deep Equilibrium Net on the stochastic Brock-Mirman model and
produce the validation figure used in lecture 02 (02_DEQN.tex).

Model
-----
    max E sum_t beta^t log C_t   s.t.   C_t + K_{t+1} = z_t K_t^alpha,  delta = 1
    log z_{t+1} = rho * log z_t + sigma_z * eps_{t+1},  eps ~ N(0,1)

Closed form:  K_{t+1} = s* z_t K_t^alpha  with  s* = alpha * beta.

Network
-------
    (K_t, log z_t) -> standardize -> 2 x 32 tanh -> sigmoid head -> s_t in (0,1)

Loss
----
    Expectation INSIDE, square OUTSIDE:

        R(x) = 1 - beta * C_t * sum_q w_q * alpha z'(eps_q) K'^(alpha-1) / C'(eps_q)
        J    = mean_i R(x_i)^2

    with 5-node Gauss-Hermite quadrature, rescaled from the physicists'
    convention returned by numpy to the probabilists' convention:
        eps_q = sqrt(2) * x_q ,  w_q = wt_q / sqrt(pi).

Outputs
-------
    bm_validation.pdf : left panel  = learned savings rate vs the closed form
                        right panel = histogram of log10 |e_REE| on a test path
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- parameters
ALPHA, BETA = 0.36, 0.96
RHO, SIGMA_Z = 0.80, 0.02
S_STAR = ALPHA * BETA          # 0.3456

N_QUAD = 5
EPISODES = 300
SIM_LEN = 2000
BURN_IN = 200
STEPS_PER_EPISODE = 30
BATCH = 256
LR = 1e-3

# ------------------------------------------------- Gauss-Hermite, probabilists'
_x, _w = np.polynomial.hermite.hermgauss(N_QUAD)   # physicists': weight e^{-x^2}
EPS_Q = torch.tensor(np.sqrt(2.0) * _x, dtype=torch.float64)
W_Q = torch.tensor(_w / np.sqrt(np.pi), dtype=torch.float64)
assert abs(W_Q.sum().item() - 1.0) < 1e-12
assert abs((W_Q * EPS_Q**2).sum().item() - 1.0) < 1e-10


# ------------------------------------------------------------------- network
class SavingsNet(nn.Module):
    """Maps (K, log z) to a savings rate in (0,1) through a sigmoid head."""

    def __init__(self, mu, sd, width=32):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.body = nn.Sequential(
            nn.Linear(2, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, K, logz):
        x = torch.stack([K, logz], dim=-1)
        x = (x - self.mu) / self.sd
        return torch.sigmoid(self.body(x)).squeeze(-1)


def steady_state_K():
    """Deterministic steady state of capital at z = 1."""
    return (ALPHA * BETA) ** (1.0 / (1.0 - ALPHA))


def euler_residual(net, K, logz):
    """R(x) with the conditional expectation taken inside, then squared outside."""
    z = torch.exp(logz)
    Y = z * K**ALPHA
    s = net(K, logz)
    C = (1.0 - s) * Y
    Kp = s * Y

    # one column per quadrature node
    logz_p = RHO * logz.unsqueeze(-1) + SIGMA_Z * EPS_Q            # (n, Q)
    z_p = torch.exp(logz_p)
    Kp_rep = Kp.unsqueeze(-1).expand_as(z_p)

    s_p = net(Kp_rep.reshape(-1), logz_p.reshape(-1)).reshape(z_p.shape)
    C_p = (1.0 - s_p) * z_p * Kp_rep**ALPHA
    mpk = ALPHA * z_p * Kp_rep ** (ALPHA - 1.0)

    expectation = (W_Q * mpk / C_p).sum(dim=-1)                     # E_t[.]
    return 1.0 - BETA * C * expectation


def simulate(net, n, K0=None, logz0=0.0):
    """Simulate a path under the current policy. States are detached."""
    with torch.no_grad():
        K = torch.full((1,), K0 if K0 is not None else steady_state_K(),
                       dtype=torch.float64)
        logz = torch.full((1,), logz0, dtype=torch.float64)
        Ks, logzs = [], []
        for _ in range(n):
            Ks.append(K.clone())
            logzs.append(logz.clone())
            s = net(K, logz)
            K = s * torch.exp(logz) * K**ALPHA
            logz = RHO * logz + SIGMA_Z * torch.randn(1, dtype=torch.float64)
        return torch.cat(Ks), torch.cat(logzs)


def main():
    torch.set_default_dtype(torch.float64)

    Kss = steady_state_K()
    mu = torch.tensor([Kss, 0.0])
    sd = torch.tensor([0.25 * Kss, SIGMA_Z / np.sqrt(1 - RHO**2)])
    net = SavingsNet(mu, sd)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    for ep in range(EPISODES):
        K, logz = simulate(net, SIM_LEN)
        K, logz = K[BURN_IN:], logz[BURN_IN:]          # detached by construction
        for _ in range(STEPS_PER_EPISODE):
            idx = torch.randint(0, K.numel(), (BATCH,))
            loss = (euler_residual(net, K[idx], logz[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 50 == 0:
            print(f"episode {ep+1:4d}   loss {loss.item():.3e}")

    # ------------------------------------------------------------ evaluation
    K_test, logz_test = simulate(net, 4000)
    K_test, logz_test = K_test[BURN_IN:], logz_test[BURN_IN:]

    with torch.no_grad():
        # e_REE in consumption units on the test path
        z = torch.exp(logz_test)
        Y = z * K_test**ALPHA
        s = net(K_test, logz_test)
        C = (1.0 - s) * Y
        R = euler_residual(net, K_test, logz_test)
        # R = 1 - C / Chat  =>  e_REE = Chat/C - 1 = R / (1 - R)
        e_ree = R / (1.0 - R)

        # policy over the ergodic range of capital, at z = 1
        k_lo, k_hi = K_test.min().item(), K_test.max().item()
        pad = 0.15 * (k_hi - k_lo)
        Kgrid = torch.linspace(k_lo - pad, k_hi + pad, 200)
        s_grid = net(Kgrid, torch.zeros_like(Kgrid))

    max_dev = (s_grid - S_STAR).abs().max().item()
    med_err = np.median(np.abs(e_ree.numpy()))
    print(f"\ns* = {S_STAR:.4f}")
    print(f"max |s(K) - s*| over the plotted range : {max_dev:.2e}")
    print(f"median |e_REE| on the test path        : {med_err:.2e}")
    print(f"log10 median                           : {np.log10(med_err):.2f}")

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.3))

    ax1.axhline(S_STAR, color="#8b0000", lw=2.2, ls="--",
                label=rf"closed form $s^\star=\alpha\beta={S_STAR:.4f}$")
    ax1.plot(Kgrid.numpy(), s_grid.numpy(), color="#000080", lw=2.0,
             label="DEQN policy")
    ax1.axvspan(k_lo, k_hi, color="#4682b4", alpha=0.12,
                label="ergodic range of $K$")
    ax1.set_xlabel(r"capital $K_t$")
    ax1.set_ylabel(r"savings rate $s_t$")
    ax1.set_ylim(S_STAR - 0.02, S_STAR + 0.02)
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_title(rf"max deviation $={max_dev:.1e}$")

    logabs = np.log10(np.abs(e_ree.numpy()) + 1e-16)
    ax2.hist(logabs, bins=45, color="#009e73", edgecolor="white", lw=0.4)
    ax2.axvline(np.log10(med_err), color="#8b0000", lw=2.0, ls="--",
                label=rf"median $={med_err:.1e}$")
    ax2.set_xlabel(r"$\log_{10}\,|e_{\mathrm{REE}}|$  (consumption units)")
    ax2.set_ylabel("test states")
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_title("Euler errors, independent test path")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = "bm_validation.pdf"
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
