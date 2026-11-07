import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """Same toy dynamics as SARSA."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def eps_greedy(Q, s, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[s][a])

def greedy_pi(Q):
    """pi(s) = argmax_a Q(s,a) (algorithm output)."""
    return {s: max(ACTIONS, key=lambda a: Q[s][a]) for s in STATES}

def q_learning(eps=0.1, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize Q(s,a) arbitrarily (algorithm step 1); terminal bootstrap is 0 below
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    n_ep = 0
    while True:  # for each episode
        s = "A"  # Initialize S_0
        while s not in TERMINALS:
            a = eps_greedy(Q, s, eps)  # Choose A_t eps-greedy in S_t (behavior)
            s_next, r = step(s, a)  # Take A_t; observe R_{t+1}, S_{t+1}
            q_next = (
                0.0
                if s_next in TERMINALS
                else max(Q[s_next][a2] for a2 in ACTIONS)
            )
            Q[s][a] += alpha * (r + gamma * q_next - Q[s][a])
            s = s_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, greedy_pi(Q)

if __name__ == "__main__":
    Q, pi = q_learning(eps=0.1, alpha=0.1, gamma=0.9)
    print("Q-learning Q:", Q)
    print("Q-learning pi:", pi)
