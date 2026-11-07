import numpy as np

def policy_evaluation(P_model, policy, V, n_states, terminals, gamma=0.99, eps=1e-8):
    """Iterative policy evaluation for a deterministic policy (updates V in place)."""
    terminals = set(terminals)
    while True:
        delta = 0.0
        for s in range(n_states):
            if s in terminals:
                continue
            v = V[s]
            a = policy[s]  # follow pi only -- no max over actions
            V[s] = 0.0
            for prob, s_next, reward in P_model[s][a]:
                V[s] += prob * (reward + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
        if delta < eps:
            break
    return V

def policy_improvement(P_model, V, policy, n_states, n_actions, terminals, gamma=0.99):
    """Greedy improvement from V. Returns True if the policy is unchanged (converged)."""
    terminals = set(terminals)
    converged = True
    for s in range(n_states):
        if s in terminals:
            continue
        old = policy[s]
        best_a, best_score = 0, -np.inf
        for a in range(n_actions):
            score = 0.0
            for prob, s_next, reward in P_model[s][a]:
                score += prob * (reward + gamma * V[s_next])
            if score > best_score:
                best_score, best_a = score, a
        policy[s] = best_a
        if best_a != old:
            converged = False
    return converged

def policy_iteration(P_model, n_states, n_actions, terminals, gamma=0.99, eps=1e-8):
    terminals = set(terminals)
    # Arbitrary initial policy on non-terminals (random actions).
    policy = np.random.randint(0, n_actions, size=n_states)
    for s in terminals:
        policy[s] = -1

    # Arbitrary V on non-terminals; terminals must be 0.
    V = np.random.randn(n_states).astype(np.float64)
    for s in terminals:
        V[s] = 0.0

    while True:
        policy_evaluation(P_model, policy, V, n_states, terminals, gamma, eps)
        if policy_improvement(
            P_model, V, policy, n_states, n_actions, terminals, gamma
        ):
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
    V_opt, policy_opt = policy_iteration(P_model, n_states, n_actions, terminals)
    print("Optimal State Values:", V_opt)
    print("Optimal Policy Actions:", policy_opt)
