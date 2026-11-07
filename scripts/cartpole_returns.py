"""CartPole DQN curves for Chapter 14 (Figure 14.2 / cartpole_returns.png).

Uses the book listing in tex/code/ch14/cartpole_dqn.py, then saves the
same three-panel plot the chapter includes.
"""

import math
import os
import random
import shutil
import sys

import matplotlib.pyplot as plt
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tex", "code", "ch14"))

from cartpole_dqn import X_MAX, dqn_train  # noqa: E402

OUT_NAME = "cartpole_returns.png"
OUT_DIRS = (
    os.path.join(ROOT, "figures"),
    os.path.join(ROOT, "tex", "figures"),
)


def mean20(seq):
    w = 20
    return [
        sum(seq[max(0, i - w + 1) : i + 1]) / min(i + 1, w)
        for i in range(len(seq))
    ]


def main():
    random.seed(0)
    torch.manual_seed(0)
    _, returns, max_xs, max_phis = dqn_train()

    ep = range(1, len(returns) + 1)
    deg = 180.0 / math.pi
    max_phis_deg = [t * deg for t in max_phis]

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

    primary = os.path.join(OUT_DIRS[0], OUT_NAME)
    os.makedirs(OUT_DIRS[0], exist_ok=True)
    fig.savefig(primary, dpi=150)
    plt.close(fig)

    for d in OUT_DIRS[1:]:
        os.makedirs(d, exist_ok=True)
        shutil.copy2(primary, os.path.join(d, OUT_NAME))

    print("saved", primary)
    print("last-20 train mean:", sum(returns[-20:]) / 20)


if __name__ == "__main__":
    main()
