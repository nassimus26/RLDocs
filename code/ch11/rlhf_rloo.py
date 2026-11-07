import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PROMPTS = ["2+2", "3+1"]
VOCAB = ["it", "is", "equals", "4", "5"]  # token ids 0..4
T = 3  # tokens per response
N_IN = len(PROMPTS) + T * len(VOCAB)
ANS = VOCAB.index("4")

def toks(*words):
    """Map word tokens to ids, e.g. toks("it", "is", "4") -> [0, 1, 3]."""
    return [VOCAB.index(w) for w in words]

PREFS = [  # Section 11.2: (x, y_w, y_l)
    ("2+2", toks("it", "is", "4"), toks("it", "is", "5")),
    ("2+2", toks("it", "equals", "4"), toks("it", "is", "5")),
    ("2+2", toks("it", "is", "4"), toks("it", "equals", "5")),
    ("2+2", toks("it", "equals", "4"), toks("it", "equals", "5")),
    ("2+2", toks("it", "is", "4"), toks("5", "is", "it")),
    ("2+2", toks("equals", "it", "4"), toks("is", "it", "5")),
    ("3+1", toks("it", "is", "4"), toks("it", "is", "5")),
    ("3+1", toks("it", "equals", "4"), toks("it", "is", "5")),
    ("3+1", toks("it", "is", "4"), toks("it", "equals", "5")),
    ("3+1", toks("it", "equals", "4"), toks("it", "equals", "5")),
    ("3+1", toks("it", "is", "4"), toks("5", "equals", "it")),
    ("3+1", toks("equals", "is", "4"), toks("is", "equals", "5")),
]

def x_state(x, y):
    """One-hot vector for state (x, y_1..y_t): prompt, then each token so far."""
    v = torch.zeros(N_IN)
    v[PROMPTS.index(x)] = 1.0
    for j, tok in enumerate(y):
        v[len(PROMPTS) + j * len(VOCAB) + tok] = 1.0
    return v

def make_h():
    """Preference network h(s,*;theta): one logit per next token. Its weights are theta."""
    return nn.Sequential(nn.Linear(N_IN, 64), nn.ReLU(), nn.Linear(64, len(VOCAB)))

def make_r():
    """Reward model r_phi(x,y): one output for a full response. Its weights are phi."""
    return nn.Sequential(nn.Linear(N_IN, 64), nn.ReLU(), nn.Linear(64, 1))

def r_phi(r, x, y):
    """r_phi(x,y)."""
    return r(x_state(x, y)).squeeze()

def sample_response(h, x):
    """y ? pi_theta(*|x), one token at a time: y_{t+1} ? pi_theta(*|x, y_1..y_t)."""
    y = []
    with torch.no_grad():
        for _ in range(T):
            y.append(int(Categorical(logits=h(x_state(x, y))).sample().item()))
    return y

def log_pi(h, x, y):
    """log pi_theta(y|x) = sum_t log pi_theta(y_{t+1}|x, y_1..y_t)  (11.1)."""
    return sum(
        Categorical(logits=h(x_state(x, y[:t]))).log_prob(torch.tensor(y[t]))
        for t in range(T)
    )

def train_reward_model(data, alpha_phi=0.05, epochs=40):
    # 2. phi <- phi ? ?_phi ?_phi [?log ?(r_phi(x,y_w) ? r_phi(x,y_l))]: SGD on L(phi) (11.3)
    r = make_r()
    opt = torch.optim.SGD(r.parameters(), lr=alpha_phi)
    for _ in range(epochs):
        for x, y_w, y_l in data:
            loss = -F.logsigmoid(r_phi(r, x, y_w) - r_phi(r, x, y_l))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return r

def rlhf(pi_ref, r, beta=0.1, K=4, alpha_theta=0.05, max_iters=1000):
    # 3. Initialize theta from pi_ref
    h = copy.deepcopy(pi_ref)
    opt = torch.optim.SGD(h.parameters(), lr=alpha_theta)
    for _ in range(max_iters):  # 4. Loop for each iteration
        x = random.choice(PROMPTS)
        ys = [sample_response(h, x) for _ in range(K)]  # y_1..y_K ? pi_theta(*|x)
        logps = [log_pi(h, x, y) for y in ys]  # log pi_theta(y_k|x)

        # R_k <- r_phi(x,y_k) ? ? (log pi_theta(y_k|x) ? log pi_ref(y_k|x))  (11.4)
        with torch.no_grad():
            R = [r_phi(r, x, y).item() - beta * (lp.item() - log_pi(pi_ref, x, y).item())
                 for y, lp in zip(ys, logps)]
        # b_k <- average of the other K?1 rewards
        b = [(sum(R) - R[k]) / (K - 1) for k in range(K)]

        # theta <- theta + ?_theta (1/K) sum_k (R_k ? b_k) ?_theta log pi_theta(y_k|x): descent on L using (11.6)-(11.7)
        loss = -sum((R[k] - b[k]) * logps[k] for k in range(K)) / K
        opt.zero_grad()
        loss.backward()
        opt.step()
    return h  # 5. Return pi_theta

def avg_answer_hits(h, n=200):
    """Average count of the answer token '4' in sampled responses."""
    return sum(
        sample_response(h, x).count(ANS) for x in PROMPTS for _ in range(n)
    ) / (len(PROMPTS) * n)

if __name__ == "__main__":
    pi_ref = make_h()
    r = train_reward_model(PREFS)
    h = rlhf(pi_ref, r)
    print("human score under pi_ref:", avg_answer_hits(pi_ref))
    print("human score under pi_theta:  ", avg_answer_hits(h))
