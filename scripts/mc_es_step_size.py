"""MC ES on a letter-state chain (Chapter 4 style): 1/N vs fixed alpha."""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]
TERMINALS = ["I", "J"]
ACTIONS = (0, 1, 2, 3)
FORWARD = {s: nxt for s, nxt in zip(STATES, STATES[1:] + ["I"])}
BACKWARD = {s: prv for s, prv in zip(STATES, ["A"] + STATES[:-1])}
TARGET_S = "A"
TARGET_A = 0
N_EPISODES = 3000
GAMMA = 0.9
ALPHA_FIXED = 0.1
NOISE_STD = 0.2
MAX_STEPS = 100
ROLL = 40

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIRS = (
    os.path.join(ROOT, "figures"),
    os.path.join(ROOT, "tex", "figures"),
)
OUT_DIR = OUT_DIRS[0]  # primary save location


def step(s, a, rng):
    if a == 0:
        s_next = FORWARD[s]
    elif a == 1:
        s_next = BACKWARD[s]
    elif a == 2:
        s_next = "J"
    else:
        s_next = s
    if s_next == "I":
        reward = 1.0
    elif s_next == "J":
        reward = -1.0
    else:
        reward = -0.05
    reward += rng.normal(0.0, NOISE_STD)
    return s_next, reward


def generate_episode(pi, rng):
    s = STATES[int(rng.integers(len(STATES)))]
    a = int(rng.integers(len(ACTIONS)))
    episode = []
    for _ in range(MAX_STEPS):
        s_next, r = step(s, a, rng)
        episode.append((s, a, r))
        s = s_next
        if s in TERMINALS:
            break
        a = pi[s]
    return episode


def greedy_action(Q, s):
    return int(np.argmax(Q[s]))


def run_mc_es(use_one_over_n, seed):
    rng = np.random.default_rng(seed)
    Q = {s: np.zeros(len(ACTIONS)) for s in STATES}
    N = {s: np.zeros(len(ACTIONS), dtype=int) for s in STATES}
    pi = {s: greedy_action(Q, s) for s in STATES}

    q_hist = np.zeros(N_EPISODES)
    flips_hist = np.zeros(N_EPISODES, dtype=int)

    for ep in range(N_EPISODES):
        episode = generate_episode(pi, rng)
        G = 0.0
        pairs = [(s, a) for s, a, _ in episode]
        seen = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + GAMMA * G
            if (s, a) in seen:
                continue
            if (s, a) in pairs[:t]:
                continue
            seen.add((s, a))
            N[s][a] += 1
            alpha = (1.0 / N[s][a]) if use_one_over_n else ALPHA_FIXED
            Q[s][a] += alpha * (G - Q[s][a])

        flips = 0
        visited = []
        for s, _, _ in episode:
            if s not in visited:
                visited.append(s)
        for s in visited:
            a_star = greedy_action(Q, s)
            if a_star != pi[s]:
                flips += 1
                pi[s] = a_star

        q_hist[ep] = Q[TARGET_S][TARGET_A]
        flips_hist[ep] = flips

    return q_hist, flips_hist


def rolling_mean(x, window):
    k = np.ones(window) / window
    return np.convolve(x, k, mode="valid")


def main():
    q_avg, flips_avg = run_mc_es(True, seed=0)
    q_fix, flips_fix = run_mc_es(False, seed=1)

    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
    episodes = np.arange(1, N_EPISODES + 1)
    ep_roll = np.arange(ROLL, N_EPISODES + 1)

    fig, (ax_q, ax_pi) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax_q.plot(episodes, q_avg, color="#1f77b4", lw=1.2, label=r"$1/N$")
    ax_q.plot(episodes, q_fix, color="#d62728", lw=1.2, alpha=0.85, label=r"$\alpha=0.1$")
    ax_q.set_xlabel("Episode")
    ax_q.set_ylabel(r"$Q(A,0)$")
    ax_q.legend(frameon=False)
    ax_pi.plot(ep_roll, rolling_mean(flips_avg, ROLL), color="#1f77b4", lw=1.4, label=r"$1/N$")
    ax_pi.plot(ep_roll, rolling_mean(flips_fix, ROLL), color="#d62728", lw=1.4, alpha=0.85, label=r"$\alpha=0.1$")
    ax_pi.set_xlabel("Episode")
    ax_pi.set_ylabel(f"Policy flips ({ROLL}-episode mean)")
    ax_pi.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT_DIRS[0], "mc_es_step_size.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    for d in OUT_DIRS[1:]:
        shutil.copy2(out, os.path.join(d, "mc_es_step_size.png"))

    late = slice(N_EPISODES - 500, None)
    print("saved", out)
    print("late-500 mean |dQ| 1/N:", float(np.mean(np.abs(np.diff(q_avg[late])))))
    print("late-500 mean |dQ| fix:", float(np.mean(np.abs(np.diff(q_fix[late])))))
    print("late-500 mean flips 1/N:", float(np.mean(flips_avg[late])))
    print("late-500 mean flips fix:", float(np.mean(flips_fix[late])))


if __name__ == "__main__":
    main()
