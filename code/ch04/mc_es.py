# STATES, TERMINALS, ACTIONS, step: Section 4.2

def generate_episode(pi):
    """Exploring starts: choose (S_0, A_0) uniform, then follow pi.
    Returns list of (S_t, A_t, R_{t+1}).
    """
    s = random.choice(STATES)  # S_0
    a = random.choice(ACTIONS)  # A_0 (not from pi)
    episode = []
    while s not in TERMINALS:
        s_next, r = step(s, a)
        episode.append((s, a, r))
        s = s_next
        if s not in TERMINALS:
            a = pi[s]  # follow pi from S_1 onward
    return episode

def mc_es(gamma=0.99, max_episodes=500):
    # Initialize Q(s,a) arbitrarily, N(s,a)<-0, pi arbitrarily (algorithm step 1)
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    N = {s: {a: 0 for a in ACTIONS} for s in STATES}
    pi = {s: random.choice(ACTIONS) for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        G = 0.0
        seen = set()
        for t in range(len(episode) - 1, -1, -1):  # t = T-1, ..., 0
            s, a, r = episode[t]
            G = r + gamma * G
            if (s, a) in seen:  # first visit
                continue
            seen.add((s, a))
            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]
        # For each s appearing in the episode: pi(s)<-argmax_a Q(s,a)
        for s, a, r in episode:
            pi[s] = max(ACTIONS, key=lambda aa: Q[s][aa])
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, pi

if __name__ == "__main__":
    Q, pi = mc_es(gamma=0.9)
    print("MC ES Q:", Q)
    print("MC ES pi:", pi)
