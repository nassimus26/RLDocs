def mc_prediction_first_visit(pi, gamma=0.99, max_episodes=500):
    # Initialize V(s) arbitrarily, N(s)<-0 (algorithm step 1)
    V = {s: random.random() for s in STATES}
    N = {s: 0 for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        states = [s for s, a, r in episode]
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):  # t = T-1, ..., 0
            s, a, r = episode[t]
            G = r + gamma * G
            if s not in states[:t]:  # first visit
                N[s] += 1
                V[s] += (G - V[s]) / N[s]
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

def mc_prediction_every_visit(pi, gamma=0.99, max_episodes=500):
    V = {s: random.random() for s in STATES}
    N = {s: 0 for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G
            N[s] += 1
            V[s] += (G - V[s]) / N[s]
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    print("First-visit:", mc_prediction_first_visit(pi, gamma=0.9))
    print("Every-visit:", mc_prediction_every_visit(pi, gamma=0.9))
