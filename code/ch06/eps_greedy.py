import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def eps_greedy(Q, s, eps):
    """Sample A ~ pi(*|s) from (6.7)."""
    a_star = max(ACTIONS, key=lambda a: Q[s][a])
    n = len(ACTIONS)
    probs = [
        1 - eps + eps / n if a == a_star else eps / n
        for a in ACTIONS
    ]
    return random.choices(ACTIONS, weights=probs, k=1)[0]

def eps_schedule(episode, eps0, lam):
    """eps_k = eps_0 * ?^k for episode k (6.8)."""
    return eps0 * (lam ** episode)

if __name__ == "__main__":
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    eps0 = 1.0
    lam = 0.99
    for episode in (0, 100):
        eps = eps_schedule(episode, eps0, lam)  # (6.8)
        a = eps_greedy(Q, "A", eps)  # (6.7) choose A_t
        print(f"episode {episode}, eps={eps:.4f}, A={a}")
