import random
from collections import deque
import torch
import torch.nn as nn

def make_q(state_dim, n_actions):
    # Maps state (size state_dim) -> Q-values, one per action (size n_actions).
    # 64 is the hidden-layer width (chosen capacity; not part of the MDP).
    return nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        nn.Linear(64, n_actions),
    )

def eps_greedy(theta, state, n_actions, eps):
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        q = theta(torch.tensor(state, dtype=torch.float32))
        return int(q.argmax().item())

def dqn_train(
    env_reset,
    env_step,
    state_dim,
    n_actions,
    episodes=50,
    buffer_capacity=10000,
    batch_size=32,
    gamma=0.99,
    eps=0.1,
    alpha=1e-3,
    target_copy_period=100,
):
    # 1. Initialize theta, theta^- <- theta, empty D, k <- 0
    theta = make_q(state_dim, n_actions)           # online weights theta
    theta_ = make_q(state_dim, n_actions)          # target weights theta^-
    theta_.load_state_dict(theta.state_dict())      # theta^- <- theta
    opt = torch.optim.SGD(theta.parameters(), lr=alpha)  # step size ?
    D = deque(maxlen=buffer_capacity)              # empty replay buffer D, capacity N
    k = 0                                          # global step counter

    # 2. Loop for episode = 1..M
    for _ in range(episodes):
        s = env_reset()
        done = False
        while not done:
            a = eps_greedy(theta, s, n_actions, eps)
            s_next, r, done = env_step(s, a)
            d = 1.0 if done else 0.0
            D.append((s, a, r, s_next, d))

            if len(D) >= batch_size:
                batch = random.sample(D, batch_size)
                states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.int64)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                q_sa = theta(states)[range(batch_size), actions]
                with torch.no_grad():
                    # y_j <- r_j + gamma(1-d_j) max_a' Q(s'_j, a'; theta^-)
                    y = rewards + gamma * (1.0 - dones) * theta_(next_states).max(dim=1).values
                # theta <- theta + (?/B) sum_j (y_j - Q) ?_theta Q
                loss = ((y - q_sa) ** 2).mean()  # mean of (y_j - Q)^2; matches (8.4) up to the constant 1/2
                opt.zero_grad()
                loss.backward()
                opt.step()

            k += 1
            if k % target_copy_period == 0:
                theta_.load_state_dict(theta.state_dict())  # theta^- <- theta

            s = s_next

    # 3. Return theta and greedy pi
    def greedy_pi(state):
        return eps_greedy(theta, state, n_actions, eps=0.0)

    return theta, greedy_pi

STATE_DIM, N_ACTIONS = 4, 2

def toy_reset():
    return [random.random() for _ in range(STATE_DIM)]

def toy_step(s, a):
    s_next = [random.random() for _ in range(STATE_DIM)]
    r = 1.0 if a == 0 else 0.0
    done = random.random() < 0.1
    return s_next, r, done

if __name__ == "__main__":
    theta, pi = dqn_train(toy_reset, toy_step, STATE_DIM, N_ACTIONS, episodes=5)
    print("greedy action at a random state:", pi(toy_reset()))
