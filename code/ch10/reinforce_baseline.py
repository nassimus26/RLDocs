import random
import torch
import torch.nn as nn
from torch.distributions import Categorical

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """One transition: next state uniform over non-terminals and terminals."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def x_s(s):
    """One-hot vector for state s (network input)."""
    x = torch.zeros(len(STATES))
    x[STATES.index(s)] = 1.0
    return x

def make_h():
    """Preference network h(s,*;theta): one logit per action. Its weights are theta."""
    return nn.Sequential(nn.Linear(len(STATES), 64), nn.ReLU(), nn.Linear(64, len(ACTIONS)))

def make_v():
    """State-value network V(s;w): one output. Its weights are w."""
    return nn.Sequential(nn.Linear(len(STATES), 64), nn.ReLU(), nn.Linear(64, 1))

def pi_theta(h, s):
    """Returns pi_theta(*|s) = softmax(h(s,*;theta)), a distribution over actions."""
    return Categorical(logits=h(x_s(s)))

def v_hat(v, s):
    """V(s;w)."""
    return v(x_s(s)).squeeze()

def generate_episode(h):
    """Roll out one episode under pi_theta: lists of S_t, log pi_theta(A_t|S_t), and R_{t+1}."""
    states, logps, rewards = [], [], []
    s = "A"  # S_0
    while s not in TERMINALS:
        dist = pi_theta(h, s)
        a = dist.sample()  # A_t ? pi_theta(*|S_t)
        s_next, r = step(s, int(a.item()))
        states.append(s)
        logps.append(dist.log_prob(a))
        rewards.append(r)
        s = s_next
    return states, logps, rewards

def reinforce_baseline(alpha_theta=0.01, alpha_w=0.01, gamma=0.99, max_episodes=500):
    # 1. Initialize theta (weights of h) and w (weights of v) randomly
    h, v = make_h(), make_v()
    opt_theta = torch.optim.SGD(h.parameters(), lr=alpha_theta)
    opt_w = torch.optim.SGD(v.parameters(), lr=alpha_w)
    n_ep = 0
    while True:  # 2. Loop for each episode
        states, logps, rewards = generate_episode(h)
        T = len(rewards)

        # For each t: G_t <- sum_{k=t}^{T-1} gamma^{k-t} R_{k+1}  (rewards[k] is R_{k+1})
        G = [sum(gamma ** (k - t) * rewards[k] for k in range(t, T)) for t in range(T)]
        V = [v_hat(v, states[t]) for t in range(T)]  # V(S_t;w) before the value update

        # w <- w + ?_w sum_t (G_t ? V(S_t;w)) ?_w V(S_t;w): one SGD step on L(w) (10.4)
        loss_w = 0.5 * sum((G[t] - V[t]) ** 2 for t in range(T))
        opt_w.zero_grad()
        loss_w.backward()
        opt_w.step()

        # theta <- theta + ?_theta sum_t gamma^t (G_t ? V(S_t;w)) ?_theta log pi_theta(A_t|S_t): one SGD step on L(theta) (10.4)
        # Uses the same V as above (not recomputed after the w step)
        loss_theta = -sum(gamma**t * (G[t] - V[t].item()) * logps[t] for t in range(T))
        opt_theta.zero_grad()
        loss_theta.backward()
        opt_theta.step()

        n_ep += 1
        if n_ep >= max_episodes:
            break
    return h, v  # 3. Return pi_theta and V

if __name__ == "__main__":
    h, v = reinforce_baseline(gamma=0.9)
    print("pi_theta(*|A):", [round(p, 3) for p in pi_theta(h, "A").probs.tolist()])
    print("V(A):", round(v_hat(v, "A").item(), 3))
