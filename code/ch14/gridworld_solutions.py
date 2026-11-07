import random
import numpy as np

N_S, N_A, GOAL = 16, 4, 15  # 4x4 cells; actions 0 up, 1 right, 2 down, 3 left

def step_logic(s, a):
    # One transition. Terminal: stay, r=0. Else clip at the wall; r=-1, or r=0 if s' is the goal.
    if s == GOAL:
        return s, 0.0, True
    row, col = divmod(s, 4)
    if a == 0:
        row = max(0, row - 1)
    elif a == 1:
        col = min(3, col + 1)
    elif a == 2:
        row = min(3, row + 1)
    else:
        col = max(0, col - 1)
    s_next = row * 4 + col
    done = s_next == GOAL
    r = 0.0 if done else -1.0
    return s_next, r, done

def build_P():
    # P[s][a] = [(prob, s', r), ...] for value iteration (here prob=1).
    P = {s: {a: [] for a in range(N_A)} for s in range(N_S)}
    for s in range(N_S):
        for a in range(N_A):
            s_next, r, _ = step_logic(s, a)
            P[s][a].append((1.0, s_next, r))
    return P

def value_iteration(P, gamma=0.99, eps=1e-8):
    # Optimality backup: V(s) <- max_a sum P [r + gamma V(s')]; then pi(s) = argmax.
    V = np.zeros(N_S)
    pi = np.full(N_S, -1, dtype=np.int64)  # -1 on the goal
    while True:
        delta = 0.0
        for s in range(N_S):
            if s == GOAL:
                continue
            v = V[s]
            best, best_a = -np.inf, 0
            for a in range(N_A):
                score = 0.0
                for p, s_next, r in P[s][a]:
                    score += p * (r + gamma * V[s_next])
                if score > best:
                    best, best_a = score, a
            V[s] = best
            pi[s] = best_a
            delta = max(delta, abs(v - V[s]))
        if delta < eps:
            break
    return V, pi

def eps_greedy(Q, s, eps):
    # How A_t is chosen (explore); not which backup is used.
    if random.random() < eps:
        return random.randrange(N_A)
    return int(np.argmax(Q[s]))

def greedy_pi(Q):
    # pi(s) = argmax_a Q(s,a). This is the object we keep after learning.
    pi = np.full(N_S, -1, dtype=np.int64)
    for s in range(N_S):
        if s != GOAL:
            pi[s] = int(np.argmax(Q[s]))
    return pi

def generate_episode(Q, eps, max_steps=100):
    # One sample trajectory from start 0: list of (S_t, A_t, R_{t+1}).
    traj = []
    s = 0
    for _ in range(max_steps):
        if s == GOAL:
            break
        a = eps_greedy(Q, s, eps)
        s_next, r, done = step_logic(s, a)
        traj.append((s, a, r))
        s = s_next
        if done:
            break
    return traj

def mc_control(eps=0.1, gamma=0.99, episodes=8000):
    # First-visit MC: target is the finished G_t; N is the visit count.
    Q = np.zeros((N_S, N_A))
    N = np.zeros((N_S, N_A))
    for _ in range(episodes):
        traj = generate_episode(Q, eps)
        G = 0.0
        seen = set()
        for s, a, r in reversed(traj):
            G = r + gamma * G
            if (s, a) in seen:
                continue
            seen.add((s, a))
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
    return Q, greedy_pi(Q)

def sarsa(eps=0.1, alpha=0.1, gamma=0.99, episodes=4000, max_steps=100):
    # On-policy TD: backup uses A' ~ pi (eps-greedy). Estimates Q^pi.
    Q = np.zeros((N_S, N_A))
    for _ in range(episodes):
        s = 0
        a = eps_greedy(Q, s, eps)
        for _ in range(max_steps):
            if s == GOAL:
                break
            s_next, r, done = step_logic(s, a)
            if done:
                q_next = 0.0  # no bootstrap from the terminal
                a_next = None
            else:
                a_next = eps_greedy(Q, s_next, eps)  # A_{t+1} from pi
                q_next = Q[s_next, a_next]
            Q[s, a] += alpha * (r + gamma * q_next - Q[s, a])
            s, a = s_next, a_next
            if done:
                break
    return Q, greedy_pi(Q)

def q_learning(eps=0.1, alpha=0.1, gamma=0.99, episodes=4000, max_steps=100):
    # Off-policy TD: backup uses max_a' Q(s', a'). Estimates Q*.
    Q = np.zeros((N_S, N_A))
    for _ in range(episodes):
        s = 0
        for _ in range(max_steps):
            if s == GOAL:
                break
            a = eps_greedy(Q, s, eps)  # behavior; not in the backup
            s_next, r, done = step_logic(s, a)
            q_next = 0.0 if done else np.max(Q[s_next])
            Q[s, a] += alpha * (r + gamma * q_next - Q[s, a])
            s = s_next
            if done:
                break
    return Q, greedy_pi(Q)

def fmt(pi):
    # Helper: print arrows for pi(s) and G on the goal. Not used for learning.
    arrows = {0: "^", 1: ">", 2: "v", 3: "<", -1: "G"}
    lines = []
    for i in range(4):
        lines.append(" ".join(arrows[int(pi[4 * i + j])] for j in range(4)))
    return "\n".join(lines)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    P = build_P()
    _, pi_vi = value_iteration(P)
    _, pi_mc = mc_control()
    _, pi_sarsa = sarsa()
    _, pi_q = q_learning()
    print("VI\n" + fmt(pi_vi))
    print("MC\n" + fmt(pi_mc))
    print("SARSA\n" + fmt(pi_sarsa))
    print("Q-learning\n" + fmt(pi_q))
