import math
import random
from collections import deque

import torch
import torch.nn as nn

GRAVITY = 9.8                          # m/s^2
MASS_CART = 1.0                        # kg
MASS_POLE = 0.1                        # kg
TOTAL_MASS = MASS_POLE + MASS_CART     # kg; cart + pole
LENGTH = 0.5                           # m; pole half-length
POLEMASS_LENGTH = MASS_POLE * LENGTH   # kg*m; used in the accelerations
FORCE_MAG = 10.0                       # N; |force| for a=0 left, a=1 right
TAU = 0.02                             # s; one physics step
PHI_MAX = 12 * 2 * math.pi / 360       # rad; fail if |phi| > 12?
X_MAX = 2.4                            # m; fail if |x| > 2.4
MAX_STEPS = 200                        # episode cutoff; G cannot exceed this
STATE_DIM, N_ACTIONS = 4, 2            # s = (x, xdot, phi, phi); two forces

def cartpole_reset():
    # s = (x, x, phi, phi); small random start near upright.
    return [(random.random() - 0.5) * 0.1 for _ in range(4)]

def cartpole_step(s, a):
    # One physics step. a=1 pushes right, a=0 left. Then test |x|, |phi|. r=1 until fail.
    x, x_dot, phi, phi_dot = s  # pole angle phi; not DQN weights theta
    force = FORCE_MAG if a == 1 else -FORCE_MAG
    cosphi = math.cos(phi)
    sinphi = math.sin(phi)
    temp = (force + POLEMASS_LENGTH * phi_dot ** 2 * sinphi) / TOTAL_MASS
    phiacc = (GRAVITY * sinphi - cosphi * temp) / (
        LENGTH * (4.0 / 3.0 - MASS_POLE * cosphi ** 2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * phiacc * cosphi / TOTAL_MASS
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    phi = phi + TAU * phi_dot
    phi_dot = phi_dot + TAU * phiacc
    s_next = [x, x_dot, phi, phi_dot]
    done = abs(x) > X_MAX or abs(phi) > PHI_MAX  # every step: off the track or pole too tilted
    r = 0.0 if done else 1.0
    return s_next, r, done

def make_q():
    # Q(s, a; theta): R^4 -> 2 action-values. No table: s is continuous.
    return nn.Sequential(
        nn.Linear(STATE_DIM, 64),
        nn.ReLU(),
        nn.Linear(64, N_ACTIONS),
    )

def eps_greedy(theta, state, eps):
    # theta is Q(*,*;theta), Chapter 8 weights -- not the pole angle phi.
    if random.random() < eps:
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        q = theta(torch.tensor(state, dtype=torch.float32))
        return int(q.argmax().item())

def dqn_train(episodes=250, gamma=0.99, alpha=1e-3, batch_size=32, C=100):
    # theta online, theta^- frozen target, D replay. Same backup as (8.2).
    theta = make_q()
    theta_ = make_q()
    theta_.load_state_dict(theta.state_dict())  # theta^- <- theta
    opt = torch.optim.Adam(theta.parameters(), lr=alpha)
    D = deque(maxlen=10000)
    k = 0
    returns = []
    max_xs, max_phis = [], []  # one |x|_max, |phi|_max per episode
    for ep in range(episodes):
        eps = max(0.05, 0.99 ** ep)  # (6.8)-style decay
        s = cartpole_reset()
        G = 0.0
        mx, mph = abs(s[0]), abs(s[2])
        for _ in range(MAX_STEPS):
            a = eps_greedy(theta, s, eps)
            s_next, r, done = cartpole_step(s, a)
            d = 1.0 if done else 0.0
            D.append((s, a, r, s_next, d))
            G += r
            if len(D) >= batch_size:
                batch = random.sample(D, batch_size)
                states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.int64)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(
                    [b[3] for b in batch], dtype=torch.float32
                )
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                q_sa = theta(states)[range(batch_size), actions]
                with torch.no_grad():
                    # y = r + gamma(1-d) max_a' Q(s', a'; theta^-)
                    y = (
                        rewards
                        + gamma
                        * (1.0 - dones)
                        * theta_(next_states).max(dim=1).values
                    )
                loss = ((y - q_sa) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            k += 1
            if k % C == 0:
                theta_.load_state_dict(theta.state_dict())  # copy theta -> theta^-
            s = s_next
            mx = max(mx, abs(s[0]))
            mph = max(mph, abs(s[2]))
            if done:
                break  # |x| or |phi| failed this step; else the loop hits MAX_STEPS
        returns.append(G)
        max_xs.append(mx)
        max_phis.append(mph)
    return theta, returns, max_xs, max_phis

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    random.seed(0)
    torch.manual_seed(0)
    theta, returns, max_xs, max_phis = dqn_train()
    print("last-20 train mean:", sum(returns[-20:]) / 20)

    def mean20(seq):
        # Helper: 20-episode mean of a per-episode series.
        w = 20
        return [
            sum(seq[max(0, i - w + 1) : i + 1]) / min(i + 1, w)
            for i in range(len(seq))
        ]

    ep = range(1, len(returns) + 1)
    deg = 180.0 / math.pi
    max_phis_deg = [t * deg for t in max_phis]
    # Helper: G, |x|_max, |phi|_max vs episode. Blue = one episode; orange = 20-episode mean.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.2, 6.6))
    axes[0].plot(ep, returns, color="#4C9BE8", lw=0.9)
    axes[0].plot(ep, mean20(returns), color="#E07A3D", lw=1.5)
    axes[0].set_ylabel("$G$")
    axes[0].set_ylim(0, 210)
    axes[1].plot(ep, max_xs, color="#4C9BE8", lw=0.9)
    axes[1].plot(ep, mean20(max_xs), color="#E07A3D", lw=1.5)
    axes[1].axhline(X_MAX, color="#888888", ls="--", lw=0.8)
    axes[1].set_ylabel(r"$|x|_{\mathrm{max}}$")
    axes[1].set_ylim(0, 2.8)
    axes[2].plot(ep, max_phis_deg, color="#4C9BE8", lw=0.9)
    axes[2].plot(ep, mean20(max_phis_deg), color="#E07A3D", lw=1.5)
    axes[2].axhline(12, color="#888888", ls="--", lw=0.8)
    axes[2].set_ylabel(r"$|\varphi|_{\mathrm{max}}$ (deg)")
    axes[2].set_xlabel("episode")
    axes[2].set_ylim(0, 16)
    axes[2].set_xlim(0, len(returns))
    axes[2].set_xticks(range(0, len(returns) + 1, 50))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
