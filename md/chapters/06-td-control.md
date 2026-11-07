# Chapter 6: TD Control: SARSA, Q-Learning, and Exploration

[← Previous: Temporal-Difference Learning](05-temporal-difference-learning.md) | [Table of contents](../main.md#chapters) | [Next: Function Approximation and the Deadly Triad](07-function-approximation-the-deadly-triad.md) →


**Control** asks which *action* to prefer from the current state, but one-step TD (5.2) only learns how good each *state* is under a fixed policy—a table of $V$ does not score actions. With a known model, DP can still choose the action
```math
\pi(s)=\arg\max_a\sum_{s',r}P(s',r\mid s,a)\bigl[r+\gamma V(s')\bigr];
```
without that $P$, the sum cannot be computed, so the $V$ table alone cannot rank actions. Model-free control therefore uses action-values $Q(s,a)$, which score each pair, so $\arg\max_a Q(s,a)$ needs no model. Monte Carlo control uses that idea, but learns $Q$ only from full returns $G_t$. This section keeps $Q$ for control and replaces that full-return backup by one-step TD, stepping $Q(S_t,A_t)$ toward a bootstrap from $Q$ at the next state. Two standard algorithms do that—**SARSA**, which is on-policy (the backup uses the next action the behavior policy actually takes), and **Q-learning**, which is off-policy (the backup uses a $\max$ over next actions).

## 6.1 SARSA

The update (5.2) steps $V(S_t)$ toward $R_{t+1}+\gamma V(S_{t+1})$, which cannot choose $\pi(S_t)$ without $P$. Replacing $V(S_t)$ by $Q(S_t,A_t)$ scores the action that was taken, by stepping $Q(S_t,A_t)$ toward $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$. Drawing both $A_t$ and $A_{t+1}$ from the same behavior $\pi$ makes the table estimate $Q^\pi$—that is **on-policy**. The method is **SARSA** (**S**tate–**A**ction–**R**eward–**S**tate–**A**ction), named for the five-tuple $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$ that the update uses. SARSA was introduced under that name by Sutton (); the learning update is a variation of the on-policy backup of , who studied it with neural function approximation. In this book we use the tabular form: one $Q(s,a)$ entry per pair, updated by (6.3) below.

The values $Q^\pi$ obey the compact Bellman expectation equation, the action-value analogue of (5.1) (<a href="#eq:B.18" data-reference-type="eqref" data-reference="eq:B.18">[eq:B.18]</a> in ):

```math
Q^\pi(s,a)=\mathbb{E}_\pi\bigl[R_{t+1}+\gamma Q^\pi(S_{t+1},A_{t+1})\mid S_t=s,A_t=a\bigr].\quad (6.1)
```

Here $A_{t+1}$ is drawn from $\pi$ at $S_{t+1}$. Observe one transition and that next action and, because $Q^\pi$ is unknown, put the current table estimate $Q$ in its place to form the one-step backup

```math
U_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}).
```

(If $S_{t+1}$ is terminal, take the $Q$ term as $0$.) The difference between that backup and the current estimate is the **TD error** on the pair—the same idea as (5.3):

```math
\delta_t=\bigl(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})\bigr) - Q(S_t,A_t).\quad (6.2)
```

Stepping $Q(S_t,A_t)$ toward $U_t$ with step-size $\alpha$ is (5.2) on pairs; the expanded and $\delta$ forms are the same update:

```math
\begin{aligned}Q(S_t,A_t)&\leftarrow Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\bigr]\\&=Q(S_t,A_t)+\alpha\,\delta_t.\end{aligned}\quad (6.3)
```

Because $A_{t+1}$ is chosen the same way as $A_t$, both from the behavior $\pi$, the actions in the backup match how we act, and the table estimates $Q^\pi$ for that same $\pi$. That is what **on-policy** means here. Recall that a **policy** is only the rule for picking an action given the state (Section 1.1), not the $Q$ table itself. Here that rule is $\varepsilon$-greedy in the **current** $Q$. With probability $1-\varepsilon$ take $\arg\max_a Q(s,a)$, otherwise pick uniformly in $\mathcal{A}$. Step 1 sets $Q(s,a)$ arbitrarily; $S_{t+1}$ comes from the environment after we take $A_t$; then $A_{t+1}$ is sampled from that $\varepsilon$-greedy $\pi$. As $Q$ improves, the same $\varepsilon$-greedy rule (now reading the new $Q$) is the updated behavior. That is evaluate–improve as in Monte Carlo control, but with a one-step TD evaluate step. The algorithm also returns $\pi'(s)=\arg\max_a Q(s,a)$. That $\pi'$ is greedy in the finished $Q$, not the $\varepsilon$-greedy $\pi$ used while learning.

Unlike Monte Carlo, SARSA does not generate the full episode first. Step 2 still loops for each episode, but the whole trajectory is not produced before updating. After $S_0$ and $A_0$, each inner step observes $(R_{t+1},S_{t+1})$, chooses $A_{t+1}$, and applies (6.3). The rest of the episode is generated only after that update, one transition at a time, until termination.

> [!NOTE]
>
> Input: $\varepsilon\in(0,1]$ (exploration); action set $\mathcal{A}$.
>
> Output: $Q\approx Q^\pi$ for the $\varepsilon$-greedy $\pi$ used while learning; greedy $\pi'(s)=\arg\max_a Q(s,a)$.
>
> 1.  Initialize $Q(s,a)$ arbitrarily, with $Q(\text{terminal},\,\cdot\,)=0$.
>
> 2.  Loop for each episode:
>
>     - Initialize $S_0$ (starting state of the episode).
>
>     - Choose $A_0$ from $Q$ $\varepsilon$-greedy in $S_0$ (over $\mathcal{A}$).
>
>     - For $t=0,1,2,\ldots$ until termination:
>
>       - Take $A_t$, observe $R_{t+1}$, $S_{t+1}$.
>
>       - Choose $A_{t+1}$ from $Q$ $\varepsilon$-greedy in $S_{t+1}$.
>
>       - $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\bigr]$.
>
> 3.  After enough episodes, return $Q$ and $\pi'(s)=\arg\max_a Q(s,a)$.

```python
import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """One transition: next state uniform over non-terminals and terminals."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def eps_greedy(Q, s, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[s][a])

def greedy_pi(Q):
    """pi'(s) = argmax_a Q(s,a) (algorithm output)."""
    return {s: max(ACTIONS, key=lambda a: Q[s][a]) for s in STATES}

def sarsa(eps=0.1, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize Q(s,a) arbitrarily (algorithm step 1); terminal bootstrap is 0 below
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    n_ep = 0
    while True:  # for each episode
        s = "A"  # Initialize S_0
        a = eps_greedy(Q, s, eps)  # Choose A_0 eps-greedy in S_0
        while s not in TERMINALS:
            s_next, r = step(s, a)  # Take A_t; observe R_{t+1}, S_{t+1}
            if s_next in TERMINALS:
                a_next = None
                q_next = 0.0
            else:
                a_next = eps_greedy(Q, s_next, eps)  # Choose A_{t+1} eps-greedy in S_{t+1}
                q_next = Q[s_next][a_next]
            Q[s][a] += alpha * (r + gamma * q_next - Q[s][a])
            s, a = s_next, a_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, greedy_pi(Q)

if __name__ == "__main__":
    Q, pi_prime = sarsa(eps=0.1, alpha=0.1, gamma=0.9)
    print("SARSA Q:", Q)
    print("SARSA pi':", pi_prime)
```

## 6.2 Q-learning

The SARSA algorithm above returns $\pi'(s)=\arg\max_a Q(s,a)$. That $\pi'$ is greedy in the finished $Q\approx Q^\pi$; it is not the $\varepsilon$-greedy $\pi$ used while learning. $\pi^*$ is an optimal policy, $\pi^*(s)=\arg\max_a Q^*(s,a)$ (2.11). SARSA’s $\pi'$ is not $\pi^*$ unless the exploring $\pi$ was already optimal. A different goal is to learn $Q^*$ so that this greedy $\pi'$ aims at $\pi^*$ while still exploring with behavior. That method is **Q-learning** (), the same one-step TD pattern on pairs, except the bootstrap uses a $\max$ over next actions instead of the behavior’s next action. The update (6.6) below is their tabular rule.

The targets $Q^*$ obey the compact Bellman optimality equation (<a href="#eq:C.5" data-reference-type="eqref" data-reference="eq:C.5">[eq:C.5]</a> in ):

```math
Q^*(s,a)=\mathbb{E}\bigl[R_{t+1}+\gamma\max_{a'} Q^*(S_{t+1},a')\mid S_t=s,A_t=a\bigr].\quad (6.4)
```

Unlike (6.1), the continuation after $S_{t+1}$ is a $\max$ over actions, not an action drawn from a behavior $\pi$. Observe one transition and, because $Q^*$ is unknown, put the current table estimate $Q$ in its place to form the one-step backup

```math
U_t=R_{t+1}+\gamma\max_{a'} Q(S_{t+1},a').
```

(If $S_{t+1}$ is terminal, take the $\max$ term as $0$.) The **TD error** on the pair is again backup minus estimate:

```math
\delta_t=\bigl(R_{t+1}+\gamma\max_{a'} Q(S_{t+1},a')\bigr) - Q(S_t,A_t).\quad (6.5)
```

Stepping $Q(S_t,A_t)$ toward $U_t$ is the same $\alpha$-update as (6.3), with this backup:

```math
\begin{aligned}Q(S_t,A_t)&\leftarrow Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma\max_{a'} Q(S_{t+1},a')-Q(S_t,A_t)\bigr]\\&=Q(S_t,A_t)+\alpha\,\delta_t.\end{aligned}\quad (6.6)
```

Unlike SARSA, where both the current action $A_t$ and the next action $A_{t+1}$ are chosen by the same behavior policy (e.g. $\varepsilon$-greedy in $Q$), Q-learning does not put a behavior-sampled $A_{t+1}$ in the backup. The next action is whichever $a'$ currently maximizes $Q(S_{t+1},a')$. Acting may still use $\varepsilon$-greedy for $A_t$ to explore; only how the backup treats the next action differs. That is why Q-learning is **off-policy**. No importance-sampling weights are needed. The $\max$ already puts the greedy target inside the one-step backup.

As in SARSA, the full episode is not generated first. Start at $S_0$, then for each step choose behavior $A_t$, observe $(R_{t+1},S_{t+1})$, and apply (6.6). There is no stored full return.

> [!NOTE]
>
> Input: $\varepsilon\in(0,1]$ (behavior exploration); action set $\mathcal{A}$.
>
> Output: $Q\approx Q^*$; greedy $\pi(s)=\arg\max_a Q(s,a)$.
>
> 1.  Initialize $Q(s,a)$ arbitrarily, with $Q(\text{terminal},\,\cdot\,)=0$.
>
> 2.  Loop for each episode:
>
>     - Initialize $S_0$ (starting state of the episode).
>
>     - For $t=0,1,2,\ldots$ until termination:
>
>       - Choose $A_t$ from $Q$ $\varepsilon$-greedy in $S_t$ (behavior; over $\mathcal{A}$).
>
>       - Take $A_t$, observe $R_{t+1}$, $S_{t+1}$.
>
>       - $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma\max_{a'} Q(S_{t+1},a')-Q(S_t,A_t)\bigr]$.
>
> 3.  After enough episodes, return $Q$ and $\pi(s)=\arg\max_a Q(s,a)$.

```python
import random

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """Same toy dynamics as SARSA."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def eps_greedy(Q, s, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[s][a])

def greedy_pi(Q):
    """pi(s) = argmax_a Q(s,a) (algorithm output)."""
    return {s: max(ACTIONS, key=lambda a: Q[s][a]) for s in STATES}

def q_learning(eps=0.1, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize Q(s,a) arbitrarily (algorithm step 1); terminal bootstrap is 0 below
    Q = {s: {a: random.random() for a in ACTIONS} for s in STATES}
    n_ep = 0
    while True:  # for each episode
        s = "A"  # Initialize S_0
        while s not in TERMINALS:
            a = eps_greedy(Q, s, eps)  # Choose A_t eps-greedy in S_t (behavior)
            s_next, r = step(s, a)  # Take A_t; observe R_{t+1}, S_{t+1}
            q_next = (
                0.0
                if s_next in TERMINALS
                else max(Q[s_next][a2] for a2 in ACTIONS)
            )
            Q[s][a] += alpha * (r + gamma * q_next - Q[s][a])
            s = s_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return Q, greedy_pi(Q)

if __name__ == "__main__":
    Q, pi = q_learning(eps=0.1, alpha=0.1, gamma=0.9)
    print("Q-learning Q:", Q)
    print("Q-learning pi:", pi)
```

## 6.3 $\varepsilon$-greedy

SARSA and Q-learning choose $A_t$ $\varepsilon$-greedy in the current $Q$ (SARSA also chooses $A_{t+1}$ that way). That rule is formulated here in mathematical form as a distribution $\pi(a\mid s)$ over $\mathcal{A}$. It is essentially the same choice as above, only written as (6.7), with an optional schedule (6.8) that shrinks $\varepsilon$. **Exploit** takes $\arg\max_a Q(s,a)$, so the action uses the current table. **Explore** draws uniformly from $\mathcal{A}$, so other pairs $(s,a)$ are visited and their $Q$ can update. $Q$ is updated only at the pair that was taken, so both are needed. With probability $1-\varepsilon$ exploit, and with probability $\varepsilon$ explore.

At state $s$, write $A^*=\arg\max_a Q(s,a)$ (break ties by any fixed rule). Because $A^*$ is still eligible on the uniform draw, its total probability is the greedy branch plus one share of that draw,

```math
\pi(A^*\mid s)=(1-\varepsilon)+\frac{\varepsilon}{\lvert\mathcal{A}\rvert}.
```

Every other action appears only on the uniform draw, each with probability $\varepsilon/\lvert\mathcal{A}\rvert$. For $\varepsilon\in(0,1]$,

```math
\pi(a\mid s)=\begin{cases}1-\varepsilon+\dfrac{\varepsilon}{\lvert\mathcal{A}\rvert},&\text{if }a=A^*,\\dfrac{\varepsilon}{\lvert\mathcal{A}\rvert},&\text{if }a\neq A^*.\end{cases}\quad (6.7)
```

When $\varepsilon=0$, (6.7) is greedy. When $\varepsilon=1$, it is uniform. A fixed $\varepsilon>0$ explores forever, which helps coverage but keeps the behavior away from the greedy $\pi^*$. A common remedy is to shrink $\varepsilon$ over episodes $k=0,1,2,\ldots$ by a geometric factor. Writing $\varepsilon_k=\varepsilon_0^k$ does not work when the start is $\varepsilon_0=1$ (then $1^k=1$ forever), and it ties the starting level to the decay speed. Keep them separate: start at $\varepsilon_0\in(0,1]$ and multiply by $\lambda^k$ with $\lambda\in(0,1)$,

```math
\varepsilon_k=\varepsilon_0\,\lambda^k.\quad (6.8)
```

Early episodes explore more; later ones concentrate on $\arg\max_a Q(s,a)$. In practice one may clip $\varepsilon_k$ below by a small floor so a little exploration remains; that floor is optional and is not part of (6.8). The same schedule can feed any method that chooses $A_t$ by (6.7)—including on-policy MC control, SARSA, and Q-learning.

The algorithm and code below are only that choice piece: given $Q$, $s$, and an $\varepsilon$ (optionally from (6.8)), return one action from (6.7). They are not a full control loop—no environment step, no $Q$ update. Plug `eps_greedy` in wherever Chapters 4–6 chose $A_t$ (or SARSA’s $A_{t+1}$) $\varepsilon$-greedy; use `eps_schedule` when $\varepsilon$ should shrink over episodes.

> [!NOTE]
>
> Input: $Q$, state $s$, $\varepsilon\in[0,1]$, action set $\mathcal{A}$.
>
> 1.  $A^*\leftarrow\arg\max_a Q(s,a)$.
>
> 2.  For each $a\in\mathcal{A}$, set $\pi(a\mid s)$ by (6.7): $1-\varepsilon+\varepsilon/\lvert\mathcal{A}\rvert$ if $a=A^*$, else $\varepsilon/\lvert\mathcal{A}\rvert$.
>
> 3.  Return a draw from $\pi(\cdot\mid s)$.

```python
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
```

## 6.4 Softmax

$\varepsilon$-greedy in (6.7) uses $Q$ only to pick the greedy action $A^*$. Every other action then shares the same mass $\varepsilon/\lvert\mathcal{A}\rvert$, so a near-best action and a terrible one are treated alike when exploring.

To rank every action by $Q$ instead, use a **Boltzmann (Gibbs)** distribution over $\mathcal{A}$: $\pi(a\mid s)\propto\exp\bigl(Q(s,a)/\tau\bigr)$, so better actions are more likely and temperature $\tau>0$ sets how sharp the ranking is. **Softmax** writes that distribution in normalized form—exponentiate each score $Q/\tau$, then divide by the sum:

```math
\pi(a\mid s)=\frac{\exp\bigl(Q(s,a)/\tau\bigr)}{\sum_{b\in\mathcal{A}}\exp\bigl(Q(s,b)/\tau\bigr)}.\quad (6.9)
```

Large $\tau$ flattens $\pi$ toward uniform; small $\tau$ concentrates on $\arg\max_a Q(s,a)$. At $\tau=1$, (6.9) is the plain softmax of $Q$. Leave $\tau$ untuned and the sharpness tracks whatever scale $Q$ happens to have, so treat $\tau$ as a hyperparameter—fixed by trying a few values, or annealed over episodes like (6.8).

The algorithm and code below only return one draw from (6.9); they are not a full control loop. Use them in place of $\varepsilon$-greedy when selecting $A_t$ (or SARSA’s $A_{t+1}$).

> [!NOTE]
>
> Input: $Q$, state $s$, temperature $\tau>0$, action set $\mathcal{A}$.
>
> 1.  For each $a\in\mathcal{A}$, set $u(a)\leftarrow\exp\bigl(Q(s,a)/\tau\bigr)$.
>
> 2.  For each $a\in\mathcal{A}$, set $\pi(a\mid s)\leftarrow u(a)/\sum_{b\in\mathcal{A}} u(b)$ as in (6.9).
>
> 3.  Return a draw from $\pi(\cdot\mid s)$.

```python
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
```

SARSA, Q-learning, (6.7), and (6.9) all use a table of $Q(s,a)$: one entry per pair, updated only when that pair is taken. That table works because $\mathcal{S}$ so far is a small fixed set. The same $s$ returns, and the same cell is updated. When $\mathcal{S}$ is large or continuous, it runs into two problems: the table does not fit, and a later observed state is almost never that same $s$, so the stored number is not used again. Chapter 7 replaces the table by a function $\hat{Q}(s,a;\mathbf{w})$ of a weight vector $\mathbf{w}$. Only those weights are stored and updated. SARSA and Q-learning keep the same backups, and (6.7) and (6.9) still choose $A_t$, now from $\hat{Q}$.

[← Previous: Temporal-Difference Learning](05-temporal-difference-learning.md) | [Table of contents](../main.md#chapters) | [Next: Function Approximation and the Deadly Triad](07-function-approximation-the-deadly-triad.md) →
