import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """One transition: next state uniform over non-terminals and terminals."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def generate_episode(pi):
    """Roll out one episode under pi: list of (S_t, A_t, R_{t+1})."""
    episode = []
    s = "A"  # S_0
    while s not in TERMINALS:
        a = pi[s]
        s_next, r = step(s, a)
        episode.append((s, a, r))
        s = s_next
    return episode

def lambda_backups(episode, V, lam, gamma):
    """U_t^? for every t (backward recursion; same as (5.6))."""
    T = len(episode)
    U = [0.0] * T
    # last step: only R_T; bootstrap at terminal is 0
    U[T - 1] = episode[T - 1][2]
    for t in range(T - 2, -1, -1):
        r = episode[t][2]
        s_next = episode[t + 1][0]
        U[t] = r + gamma * ((1 - lam) * V[s_next] + lam * U[t + 1])
    return U

def td_lambda_prediction(pi, lam=0.5, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize V(s) arbitrarily (algorithm step 1)
    V = {s: random.random() for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        U = lambda_backups(episode, V, lam, gamma)  # frozen V for every t
        for t in range(len(episode)):
            s = episode[t][0]
            V[s] += alpha * (U[t] - V[s])
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    print("TD(0):", td_lambda_prediction(pi, lam=0.0, alpha=0.1, gamma=0.9))
    print("TD(?):", td_lambda_prediction(pi, lam=0.5, alpha=0.1, gamma=0.9))
    print("TD(1):", td_lambda_prediction(pi, lam=1.0, alpha=0.1, gamma=0.9))
