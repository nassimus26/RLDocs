# STATES, TERMINALS, ACTIONS, step: Section 5.1
import random

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

def n_step_td_prediction(pi, n=2, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize V(s) arbitrarily (algorithm step 1)
    V = {s: random.random() for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        T = len(episode)
        # U_t^{(n)} for every t from the same frozen V
        U = [0.0] * T
        for t in range(T):
            for k in range(n):
                if t + k >= T:
                    break
                U[t] += (gamma ** k) * episode[t + k][2]
            if t + n < T:
                U[t] += (gamma ** n) * V[episode[t + n][0]]
            # else: episode ended; bootstrap term is 0
        for t in range(T):
            s = episode[t][0]
            V[s] += alpha * (U[t] - V[s])
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    print("1-step:", n_step_td_prediction(pi, n=1, alpha=0.1, gamma=0.9))
    print("2-step:", n_step_td_prediction(pi, n=2, alpha=0.1, gamma=0.9))
