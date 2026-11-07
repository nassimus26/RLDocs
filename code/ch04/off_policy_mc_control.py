# STATES, TERMINALS, ACTIONS, step: Section 4.2
# eps_greedy, generate_episode: Section 4.4.2

def b_prob(Q, s, a, eps):
    """b(a|s) under eps-greedy in Q (coverage for greedy pi)."""
    a_star = max(ACTIONS, key=lambda aa: Q[s][aa])
    nA = len(ACTIONS)
    if a == a_star:
        return 1.0 - eps + eps / nA
    return eps / nA

def off_policy_mc_control(eps=0.1, gamma=0.99, max_episodes=500):
    # Initialize Q(s,a) arbitrarily, C(s,a)<-0 (algorithm step 1)
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    C = {s: {a: 0.0 for a in ACTIONS} for s in STATES}
    pi = {s: max(ACTIONS, key=lambda a: Q[s][a]) for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(Q, eps)  # follow b (covers greedy pi)
        G = 0.0
        rho = 1.0
        for t in range(len(episode) - 1, -1, -1):  # t = T-1, ..., 0 while ?!=0
            if rho == 0.0:
                break
            s, a, r = episode[t]
            G = r + gamma * G
            b_sa = b_prob(Q, s, a, eps)  # b(A_t|S_t) before this step's Q update
            C[s][a] += rho
            Q[s][a] += (rho / C[s][a]) * (G - Q[s][a])
            pi[s] = max(ACTIONS, key=lambda aa: Q[s][aa])  # target pi greedy on Q
            # ? <- ? * pi(A_t|S_t) / b(A_t|S_t); pi puts 1 on its action, else 0
            pi_prob = 1.0 if a == pi[s] else 0.0
            rho *= pi_prob / b_sa
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, pi

if __name__ == "__main__":
    Q, pi = off_policy_mc_control(eps=0.1, gamma=0.9)
    print("Off-policy MC Q:", Q)
    print("Off-policy MC pi:", pi)
