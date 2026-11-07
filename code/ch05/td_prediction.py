import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """One transition: next state uniform over non-terminals and terminals."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def td_prediction(pi, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize V(s) arbitrarily (algorithm step 1); terminal bootstrap is 0 below
    V = {s: random.random() for s in STATES}
    n_ep = 0
    while True:  # for each episode
        s = "A"  # Initialize S_0
        while s not in TERMINALS:
            a = pi[s]  # A_t ~ pi(*|S_t)
            s_next, r = step(s, a)  # observe R_{t+1}, S_{t+1}
            v_next = 0.0 if s_next in TERMINALS else V[s_next]
            V[s] += alpha * (r + gamma * v_next - V[s])
            s = s_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    print("TD:", td_prediction(pi, alpha=0.1, gamma=0.9))
