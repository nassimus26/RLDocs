import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """One transition: next state uniform over non-terminals and terminals."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

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
