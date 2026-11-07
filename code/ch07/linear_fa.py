import random
import numpy as np

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]
TERMINALS = {"I", "J"}
ACTIONS = list(range(4))
D = 8  # arbitrary feature dimension
X = {(s, a): np.random.randn(D) for s in STATES for a in ACTIONS}

def step(s, a):
    """Same toy dynamics as tabular Q-learning."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def x_sa(s, a):
    return X[(s, a)]

def q_hat(w, s, a):
    return float(np.dot(w, x_sa(s, a)))

def eps_greedy(w, s, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: q_hat(w, s, a))

def greedy_pi(w):
    """pi(s) = argmax_a Q(s,a;w) (algorithm output)."""
    return {s: max(ACTIONS, key=lambda a: q_hat(w, s, a)) for s in STATES}

def full_gradient_q(eps=0.1, alpha=0.1, gamma=0.99, max_episodes=500):
    # 1. Initialize w arbitrarily
    w = np.zeros(D)
    n_ep = 0
    while True:  # 2. Loop for each episode
        s = "A"  # Initialize S
        while s not in TERMINALS:
            a = eps_greedy(w, s, eps)
            s_next, r = step(s, a)
            x = x_sa(s, a)
            if s_next in TERMINALS:
                U = r
                g_U = np.zeros(D)
            else:
                a_star = max(ACTIONS, key=lambda ap: q_hat(w, s_next, ap))
                U = r + gamma * q_hat(w, s_next, a_star)
                g_U = gamma * x_sa(s_next, a_star)
            w = w - alpha * (U - np.dot(w, x)) * (g_U - x)
            s = s_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return w, greedy_pi(w)  # 3. Return w and pi

def semi_gradient_q(eps=0.1, alpha=0.1, gamma=0.99, max_episodes=500):
    # 1. Initialize w arbitrarily
    w = np.zeros(D)
    n_ep = 0
    while True:  # 2. Loop for each episode
        s = "A"  # Initialize S
        while s not in TERMINALS:
            a = eps_greedy(w, s, eps)
            s_next, r = step(s, a)
            x = x_sa(s, a)
            if s_next in TERMINALS:
                U = r
            else:
                U = r + gamma * max(q_hat(w, s_next, ap) for ap in ACTIONS)
            w = w + alpha * (U - np.dot(w, x)) * x
            s = s_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return w, greedy_pi(w)  # 3. Return w and pi

def semi_gradient_sarsa(eps=0.1, alpha=0.1, gamma=0.99, max_episodes=500):
    # 1. Initialize w arbitrarily
    w = np.zeros(D)
    n_ep = 0
    while True:  # 2. Loop for each episode
        s = "A"
        a = eps_greedy(w, s, eps)
        while s not in TERMINALS:
            s_next, r = step(s, a)
            x = x_sa(s, a)
            if s_next in TERMINALS:
                U = r
                a_next = None
            else:
                a_next = eps_greedy(w, s_next, eps)
                U = r + gamma * q_hat(w, s_next, a_next)
            w = w + alpha * (U - np.dot(w, x)) * x
            s = s_next
            if a_next is not None:
                a = a_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return w, greedy_pi(w)  # 3. Return w and optional greedy readout

if __name__ == "__main__":
    w_full, pi_full = full_gradient_q(eps=0.1, alpha=0.1, gamma=0.9)
    w_semi, pi_semi = semi_gradient_q(eps=0.1, alpha=0.1, gamma=0.9)
    w_sarsa, pi_sarsa = semi_gradient_sarsa(eps=0.1, alpha=0.1, gamma=0.9)
    print("full-gradient Q pi:", pi_full)
    print("semi-gradient Q pi:", pi_semi)
    print("semi-gradient SARSA pi:", pi_sarsa)
