# STATES, TERMINALS, ACTIONS, step: Section 4.2
# greedy_pi: Section 4.3

def eps_greedy(Q, s, eps):
    """Sample from eps-greedy pi(*|s) on current Q."""
    if random.random() < eps:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[s][a])

def generate_episode(Q, eps):
    """Generate one episode following eps-greedy pi in Q: list of (S_t, A_t, R_{t+1})."""
    episode = []
    s = "A"  # S_0 (fixed start; no exploring starts)
    while s not in TERMINALS:
        a = eps_greedy(Q, s, eps)  # A_t ~ pi(*|S_t)
        s_next, r = step(s, a)
        episode.append((s, a, r))
        s = s_next
    return episode

def on_policy_mc_control(eps=0.1, gamma=0.99, max_episodes=500):
    # Initialize Q(s,a) arbitrarily, N(s,a)<-0 (algorithm step 1)
    # Behavior pi is eps-greedy in Q
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    N = {s: {a: 0 for a in ACTIONS} for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(Q, eps)
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
        # Improve: for each s in the episode, pi(*|s) is eps-greedy on A*=argmax Q
        # (next generate_episode samples that updated eps-greedy via Q)
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, greedy_pi(Q)

if __name__ == "__main__":
    Q, pi_prime = on_policy_mc_control(eps=0.1, gamma=0.9)
    print("On-policy MC Q:", Q)
    print("On-policy MC pi':", pi_prime)
