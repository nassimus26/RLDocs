# STATES, TERMINALS, ACTIONS, step, generate_episode: previous listing (Section 4.2)

def greedy_pi(Q):
    """pi'(s) = argmax_a Q(s,a) (algorithm output)."""
    return {s: max(ACTIONS, key=lambda a: Q[s][a]) for s in STATES}

def mc_q_prediction_first_visit(pi, gamma=0.99, max_episodes=500):
    # Initialize Q(s,a) arbitrarily, N(s,a)<-0 (algorithm step 1)
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    N = {s: {a: 0 for a in ACTIONS} for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        G = 0.0
        seen = set()
        for t in range(len(episode) - 1, -1, -1):  # t = T-1, ..., 0
            s, a, r = episode[t]
            G = r + gamma * G
            if (s, a) in seen:  # first visit only
                continue
            seen.add((s, a))
            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, greedy_pi(Q)

def mc_q_prediction_every_visit(pi, gamma=0.99, max_episodes=500):
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    N = {s: {a: 0 for a in ACTIONS} for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G
            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, greedy_pi(Q)

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    Q1, pi1 = mc_q_prediction_first_visit(pi, gamma=0.9)
    Q2, pi2 = mc_q_prediction_every_visit(pi, gamma=0.9)
    print("First-visit Q:", Q1)
    print("First-visit pi':", pi1)
    print("Every-visit Q:", Q2)
    print("Every-visit pi':", pi2)
