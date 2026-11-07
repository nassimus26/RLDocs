import math
import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def softmax(Q, s, tau):
    """Sample A ~ pi(*|s) from (6.9)."""
    logits = [Q[s][a] / tau for a in ACTIONS]
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]  # subtract m for stability
    Z = sum(exps)
    probs = [e / Z for e in exps]
    return random.choices(ACTIONS, weights=probs, k=1)[0]

if __name__ == "__main__":
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    for tau in (1.0, 0.1):
        a = softmax(Q, "A", tau)
        print(f"?={tau}, A={a}")
