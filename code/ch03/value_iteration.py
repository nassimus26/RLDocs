import numpy as np

def value_iteration(P_model, n_states, n_actions, terminals, gamma=0.99, eps=1e-8):
    """
    Parameters:
        P_model: P_model[s][a] = list of (prob, next_state, reward)
        terminals: iterable of terminal state indices (V fixed at 0)
        eps: stop when max |V change| < eps (eps in the algorithm)
    Returns:
        V: approximate V*
        policy: greedy policy on non-terminals (-1 on terminals)
    """
    terminals = set(terminals)
    # Arbitrary start on non-terminals; terminals must be 0.
    V = np.random.randn(n_states).astype(np.float64)
    for s in terminals:
        V[s] = 0.0

    policy = np.full(n_states, -1, dtype=np.int64)  # initialize policies to -1
    while True:
        delta = 0.0
        for s in range(n_states):
            if s in terminals:
                continue
            v = V[s]
            best, best_a = -np.inf, 0
            for a in range(n_actions):
                score = 0.0
                for prob, s_next, reward in P_model[s][a]:
                    score += prob * (reward + gamma * V[s_next])
                if score > best:
                    best, best_a = score, a
            V[s] = best
            policy[s] = best_a
            delta = max(delta, abs(v - V[s]))
        if delta < eps:
            break

    return V, policy

if __name__ == "__main__":
    # States: 0 = left terminal, 1 = start, 2 = goal terminal
    n_states, n_actions = 3, 2  # actions: 0 left, 1 right
    terminals = {0, 2}

    # P_model[s][a] = [(prob, next_state, reward), ...]
    P_model = {
        0: {0: [(1.0, 0, 0.0)], 1: [(1.0, 0, 0.0)]},
        1: {0: [(1.0, 0, 0.0)], 1: [(1.0, 2, 1.0)]},
        2: {0: [(1.0, 2, 0.0)], 1: [(1.0, 2, 0.0)]},
    }

    V_opt, policy_opt = value_iteration(P_model, n_states, n_actions, terminals)
    print("Optimal State Values:", V_opt)
    print("Optimal Policy Actions:", policy_opt)
