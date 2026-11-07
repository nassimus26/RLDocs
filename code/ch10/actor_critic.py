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
    """V(s;w), with V = 0 at terminal states."""
    if s in TERMINALS:
        return torch.tensor(0.0)
    return v(x_s(s)).squeeze()

def actor_critic(alpha_theta=0.01, alpha_w=0.01, gamma=0.99, max_episodes=500):
    # 1. Initialize theta (weights of h) and w (weights of v) randomly
    h, v = make_h(), make_v()
    opt_theta = torch.optim.SGD(h.parameters(), lr=alpha_theta)
    opt_w = torch.optim.SGD(v.parameters(), lr=alpha_w)
    n_ep = 0
    while True:  # 2. Loop for each episode
        s, t = "A", 0  # S_0
        while s not in TERMINALS:
            dist = pi_theta(h, s)
            a = dist.sample()  # A ? pi_theta(*|S)
            s_next, r = step(s, int(a.item()))

            # ? <- R + gamma V(S';w) ? V(S;w)  (10.7), with V before the value update
            V_s = v_hat(v, s)
            U = r + gamma * v_hat(v, s_next).item()  # target U_t, a fixed number
            delta = U - V_s.item()

            # w <- w + ?_w ? ?_w V(S;w): one SGD step on L(w) (10.10)
            loss_w = 0.5 * (U - V_s) ** 2
            opt_w.zero_grad()
            loss_w.backward()
            opt_w.step()

            # theta <- theta + ?_theta gamma^t ? ?_theta log pi_theta(A|S): one SGD step on L(theta) (10.10)
            # Uses the same ? as above (not recomputed after the w step)
            loss_theta = -(gamma**t) * delta * dist.log_prob(a)
            opt_theta.zero_grad()
            loss_theta.backward()
            opt_theta.step()

            s, t = s_next, t + 1
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return h, v  # 3. Return pi_theta and V

if __name__ == "__main__":
    h, v = actor_critic(gamma=0.9)
    print("pi_theta(*|A):", [round(p, 3) for p in pi_theta(h, "A").probs.tolist()])
    print("V(A):", round(v_hat(v, "A").item(), 3))
