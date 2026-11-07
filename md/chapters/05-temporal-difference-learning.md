# Chapter 5: Temporal-Difference Learning

[← Previous: Monte Carlo Methods](04-monte-carlo-methods.md) | [Table of contents](../main.md#chapters) | [Next: TD Control: SARSA, Q-Learning, and Exploration](06-td-control.md) →


Chapters 3 and 4 already give two ways to compute values. Neither is ideal in every setting. Dynamic Programming needs the model $P$ and a full backup over every next pair $(s',r)$; Monte Carlo needs no model and uses episode returns $G_t$. Temporal-Difference (TD) learning () sits between them: it keeps Monte Carlo’s model-free samples and Dynamic Programming’s bootstrapping, so the table can be updated from one observed pair $(R_{t+1},S_{t+1})$ rather than waiting for $G_t$. The algorithms below evaluate a given $\pi$ and return $V\approx V^\pi$. They do not replace $\pi$: that Improve step was Section 4.4 for Monte Carlo, and is Chapter 6 for TD.

Those DP limitations follow from the backup: it is a sum over $P$. In Dynamic Programming (Chapter 3), the model $P(s',r\mid s,a)$ is known, so each update is a full backup. Value iteration and the improve step of policy iteration use

```math
V(s)\leftarrow\max_a\sum_{s',r}P(s',r\mid s,a)\bigl[r+\gamma V(s')\bigr],
```

while policy evaluation uses the same sum with the fixed action $a=\pi(s)$:

```math
V(s)\leftarrow\sum_{s',r}P(s',r\mid s,\pi(s))\bigl[r+\gamma V(s')\bigr].
```

Every backup averages over all next pairs $(s',r)$ under $P$, and replaces the rest of the return by the current table entry $V(s')$—that reuse of $V$ is **bootstrapping**. No episodes are required, only sweeps over the state table. The price is dependence on $P$: if the model is unknown or the sum is infeasible, DP cannot run.

When $P$ is unavailable, Monte Carlo (Chapter 4) generates episodes under $\pi$ and never uses $P$. The return obeys (1.10), $G_t=R_{t+1}+\gamma G_{t+1}$, so $V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s]$. The average of returns from visits to $s$ is

```math
V^\pi(s)\approx\frac{1}{N}\sum_{i=1}^{N} G^{(i)}(s),
```

and the next return is folded in with step-size $\alpha$ as in (4.3)—the **Monte Carlo** update

```math
V(S_t)\leftarrow V(S_t)+\alpha\bigl(G_t-V(S_t)\bigr).
```

(Here $\alpha=1/(N{+}1)$ recovers an exact sample mean after $N$ prior samples for that state.) Because $G_t$ is built only from observed rewards, there is no sum over $P$ and no lookup of $V(S_{t+1})$. Two difficulties remain. First, $G_t$ stacks many random rewards, so the return from a given state has high variance across visits. Second, $G_t$ is known only when the episode ends, so updates wait for termination: long episodes delay learning, and continuing tasks do not fit.

Both difficulties come from using the full return as the learning target. Equivalently, the same $V^\pi$ takes the compact form, which replaces $G_{t+1}$ by $V^\pi(S_{t+1})$ (<a href="#eq:B.11" data-reference-type="eqref" data-reference="eq:B.11">[eq:B.11]</a> in ):

```math
V^\pi(s)=\mathbb{E}_\pi\bigl[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s\bigr].\quad (5.1)
```

Run the same sample-mean steps as for Monte Carlo—average the one-step backup over visits, then step with $\alpha$—but replace the unknown $V^\pi(S_{t+1})$ by the current table $V(S_{t+1})$. The result is the **TD** update

```math
V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]\quad (5.2)
```

Like Monte Carlo, (5.2) needs no $P$: the pair $(R_{t+1},S_{t+1})$ is observed from interaction. Like Dynamic Programming, it bootstraps with a table lookup at the next state. Replacing $G_{t+1}$ by $V(S_{t+1})$ avoids waiting for the episode to end and avoids the high-variance full return $G_t$. The backup $R_{t+1}+\gamma V(S_{t+1})$ still varies across visits, because $R_{t+1}$ and $S_{t+1}$ are random and $V$ is not $V^\pi$. Section 5.1 develops this **one-step TD** method (TD error and algorithm); Sections 5.2–5.3 extend it to $n$-step TD and TD($\lambda$). Control—$Q$ and an improving $\pi$—is Chapter 6.

## 5.1 One-step TD (TD(0))

The update (5.2) steps $V(S_t)$ toward the backup $R_{t+1}+\gamma V(S_{t+1})$. That backup uses only the next reward and the table at $S_{t+1}$—a **one-step** backup. The difference between that backup and the prediction is the **TD error**—the signal that tells whether $V(S_t)$ should rise or fall:

```math
\delta_t=\bigl(R_{t+1}+\gamma V(S_{t+1})\bigr) - V(S_t).\quad (5.3)
```

Equivalently, (5.2) is

```math
V(S_t)\leftarrow V(S_t)+\alpha\,\delta_t.
```

If $\delta_t>0$, then $R_{t+1}+\gamma V(S_{t+1})>V(S_t)$, so $V(S_t)$ increases; if $\delta_t<0$, it decreases. That push and pull of $V$ by $\delta_t$ is the learning rule. The lookup $V(S_{t+1})$ in the backup is the bootstrap; $V(S_t)$ is only the prediction being corrected. Both come from the table, so $V$ must be initialized before the first step (an arbitrary table; a common choice is $V\equiv 0$). If $S_{t+1}$ is terminal, take $V(S_{t+1})=0$ and the backup is just $R_{t+1}$.

The step in (5.2) is not the Monte Carlo $1/n$ of . That derivation averages $G$; with $\pi$ and $P$ fixed, $\mathbb{E}[G]=V^\pi$ does not depend on the table. Here the target is $R_{t+1}+\gamma V(S_{t+1})$, which moves with $V$ even when $\pi$ does not. Early backups use a poor $V$; later backups use a better $V$. A $1/n$ weight would fall on the early backups. A constant $\alpha\in(0,1]$ keeps weight on later backups as $V$ improves, and is the usual practical choice.

Unlike Monte Carlo, TD does **not** generate a full episode and then update. The outer loop starts another episode; the inner loop generates that episode under $\pi$ and applies (5.2) to each observed pair $(R_{t+1},S_{t+1})$. That one-step method is also called **TD(0)** (the name is explained in Section 5.3):

> [!NOTE]
>
> Input: policy $\pi$ to evaluate.
>
> Output: estimate $V\approx V^\pi$.
>
> 1.  Initialize $V(s)$ arbitrarily (e.g. $0$), with $V(\text{terminal})=0$.
>
> 2.  Loop for each episode:
>
>     - Initialize $S_0$.
>
>     - For $t=0,1,2,\ldots$ until termination:
>
>       - Take $A_t\sim\pi(\cdot\mid S_t)$, observe $R_{t+1}$, $S_{t+1}$.
>
>       - $V(S_t)\leftarrow V(S_t)+\alpha\bigl[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\bigr]$.
>
>       - $S_t\leftarrow S_{t+1}$.
>
> 3.  After enough episodes, return $V$ ($\approx V^\pi$).

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

def td_prediction(pi, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize V(s) arbitrarily (algorithm step 1); terminal bootstrap is 0 below
    V = {s: random.random() for s in STATES}
    n_ep = 0
    while True:  # for each episode
        s = "A"  # Initialize S_0
        while s not in TERMINALS:
            a = pi[s]  # A_t ~ pi(*|S_t)
            s_next, r = step(s, a)  # observe R_{t+1}, S_{t+1}
            v_next = 0.0 if s_next in TERMINALS else V[s_next]
            V[s] += alpha * (r + gamma * v_next - V[s])
            s = s_next
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    print("TD:", td_prediction(pi, alpha=0.1, gamma=0.9))
```

## 5.2 $n$-step TD

TD(0) steps toward one reward and then the table at the next state. A reward that arrives several steps later can change $V(S_t)$ only by passing through those later table entries, and if $V$ is still wrong those entries pollute the backup. $n$-step TD waits for the next $n$ rewards and bootstraps only at $S_{t+n}$, so those rewards enter the target directly. The case $n=1$ is TD(0); waiting until termination recovers the Monte Carlo return $G$. Larger $n$ uses more observed rewards and less of $V$, so less bootstrap error and faster credit, but more variance and a delay of $n$ steps. Intermediate $n$ is often better than either extreme.

The one-step backup already in (5.2) is

```math
U_t^{(1)}=R_{t+1}+\gamma V(S_{t+1}).
```

Two rewards and a bootstrap at $S_{t+2}$ start from the return identity (1.10), $G_t=R_{t+1}+\gamma G_{t+1}$. Substitute $G_{t+1}=R_{t+2}+\gamma G_{t+2}$:

```math
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 G_{t+2}.\quad (5.5)
```

$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$. Take that expectation on (5.5) and replace $G_{t+2}$ by $V^\pi(S_{t+2})$ (tower property, as in (5.1)):

```math
V^\pi(s)=\mathbb{E}_\pi\bigl[R_{t+1}+\gamma R_{t+2}+\gamma^2 V^\pi(S_{t+2})\mid S_t=s\bigr].
```

One sample $(R_{t+1},R_{t+2},S_{t+2})$ with the table $V$ in place of $V^\pi$ is the two-step backup (write $U$ for a backup; $G$ stays the true return)

```math
U_t^{(2)}=R_{t+1}+\gamma R_{t+2}+\gamma^2 V(S_{t+2}).
```

When $V=V^\pi$, $\mathbb{E}_\pi[U_t^{(2)}\mid S_t=s]=V^\pi(s)$. The update is (5.2) with this target:

```math
V(S_t)\leftarrow V(S_t)+\alpha\bigl(U_t^{(2)}-V(S_t)\bigr).
```

The same pattern with $n$ rewards and a bootstrap at $S_{t+n}$ is

```math
U_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n}),
```

```math
V(S_t)\leftarrow V(S_t)+\alpha\bigl(U_t^{(n)}-V(S_t)\bigr).
```

For $n=1$, $U_t^{(1)}$ is the backup in (5.2). If the episode ends at or before $t+n$, then $V(\text{terminal})=0$ and $U_t^{(n)}$ is the remaining return $G_t$. $U_t^{(n)}$ contains $V$, so a constant $\alpha$ is the same practical choice as in Section 5.1.

The method that uses a fixed $n$ is **$n$-step TD**. It needs the next $n$ transitions (or termination), so an episode is generated first, then $V$ is updated with $U_t^{(n)}$ from a frozen table.

> [!NOTE]
>
> Input: policy $\pi$ to evaluate; horizon $n\ge 1$.
>
> Output: estimate $V\approx V^\pi$.
>
> 1.  Initialize $V(s)$ arbitrarily (e.g. $0$), with $V(\text{terminal})=0$.
>
> 2.  Loop for each episode:
>
>     - Generate an episode following $\pi$: $S_0,A_0,R_1,\;\ldots,\;S_{T-1},A_{T-1},R_T$.
>
>     - For each $t=0,1,\ldots,T{-}1$, compute $U_t^{(n)}$ (same frozen table $V$ for every $t$).
>
>     - For each $t=0,1,\ldots,T{-}1$: $V(S_t)\leftarrow V(S_t)+\alpha\bigl(U_t^{(n)}-V(S_t)\bigr)$.
>
> 3.  After enough episodes, return $V$ ($\approx V^\pi$).

```python
# STATES, TERMINALS, ACTIONS, step: Section 5.1
import random

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

def n_step_td_prediction(pi, n=2, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize V(s) arbitrarily (algorithm step 1)
    V = {s: random.random() for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        T = len(episode)
        # U_t^{(n)} for every t from the same frozen V
        U = [0.0] * T
        for t in range(T):
            for k in range(n):
                if t + k >= T:
                    break
                U[t] += (gamma ** k) * episode[t + k][2]
            if t + n < T:
                U[t] += (gamma ** n) * V[episode[t + n][0]]
            # else: episode ended; bootstrap term is 0
        for t in range(T):
            s = episode[t][0]
            V[s] += alpha * (U[t] - V[s])
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    print("1-step:", n_step_td_prediction(pi, n=1, alpha=0.1, gamma=0.9))
    print("2-step:", n_step_td_prediction(pi, n=2, alpha=0.1, gamma=0.9))
```

Section 5.3 mixes several horizons $n$ into one backup and obtains **TD($\lambda$)**.

## 5.3 TD($\lambda$)

**Mixing backups.** In Section 5.2 the learner must pick one fixed horizon $n$: use only $U_t^{(n)}$. That commits to a single tradeoff (short $n$ bootstraps early; long $n$ waits for more rewards). A single horizon is not required. One can instead form **one** backup by averaging several horizons, then step $V$ toward that mix as in (4.3). For example, give equal weight to the 1-step, 2-step, and 3-step backups:

```math
U=\frac{1}{3}U_t^{(1)}+\frac{1}{3}U_t^{(2)}+\frac{1}{3}U_t^{(3)}=\sum_{n=1}^{3}\frac{1}{3}\,U_t^{(n)}.
```

Then

```math
V(S_t)\leftarrow V(S_t)+\alpha\bigl(U-V(S_t)\bigr)=V(S_t)+\alpha\Biggl(\sum_{n=1}^{3}\frac{1}{3}\,U_t^{(n)}-V(S_t)\Biggr).
```

More generally, pick any nonnegative weights $w_n$ that sum to one ($\sum_n w_n=1$) and set

```math
U=\sum_{n} w_n\,U_t^{(n)}.
```

The equal mix above is the special case $w_1=w_2=w_3=1/3$. The update is again

```math
V(S_t)\leftarrow V(S_t)+\alpha\Biggl(\sum_{n} w_n\,U_t^{(n)}-V(S_t)\Biggr).
```

The weights must sum to $1$ so $U$ stays a weighted **average** on the scale of a return (and of $V$). Uniform $w_n=1/N$ is fine for a **finite** list $n=1,\ldots,N$. It does **not** extend to $N\to\infty$: you cannot give every $n=1,2,3,\ldots$ the same positive weight and still have $\sum_{n=1}^{\infty} w_n=1$. A flat $1/N$ also gives the same importance to a short backup (few rewards, early use of $V$) and a long backup (many random rewards). Usually one wants short horizons to count more, and long horizons less—that leads to the geometric weights next.

**Geometric weights and $\lambda$.** Let each longer backup get a fixed fraction $\lambda\in[0,1)$ of the previous weight. The unnormalized weights are $1,\lambda,\lambda^2,\ldots$, i.e. $\lambda^{n-1}$. The first $N$ sum to

```math
1+\lambda+\cdots+\lambda^{N-1}=\frac{1-\lambda^N}{1-\lambda}\qquad(\lambda\neq 1).
```

If $N\to\infty$ and $\lambda<1$, then $\lambda^N\to 0$, so the infinite sum is $1/(1-\lambda)$. Normalizing gives

```math
w_n=(1-\lambda)\,\lambda^{n-1}.
```

As $n$ grows, $\lambda^{n-1}$ shrinks ($\lambda<1$), so $w_n$ shrinks. A short backup $U_t^{(1)}$ (one reward, then $V$) has the largest weight; a long backup $U_t^{(n)}$ (many rewards, later $V$) has less. For $\lambda=0.5$: $w_1=1/2$, $w_2=1/4$, $w_3=1/8$, $\ldots$. The resulting mix is the **$\lambda$-backup**

```math
U_t^{\lambda}=\sum_{n=1}^{\infty} w_n\,U_t^{(n)}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}U_t^{(n)},\quad (5.4)
```

and

```math
V(S_t)\leftarrow V(S_t)+\alpha\bigl(U_t^{\lambda}-V(S_t)\bigr).
```

That method is **TD($\lambda$)**. At $\lambda=0$, $U_t^{\lambda}=U_t^{(1)}$, which is the one-step backup of Section 5.1 and the $n{=}1$ case of Section 5.2, so that method is named **TD(0)**. As $\lambda\to 1$, $U_t^{\lambda}\to G_t$ (Monte Carlo)—**TD(1)**.

> [!NOTE]
>
> Equation (5.4) takes $\lambda\in[0,1)$. For every such $\lambda$ the weights still sum to $1$,
> ```math
> \sum_{n=1}^{\infty}(1-\lambda)\lambda^{n-1}=1,
> ```
> so $U_t^{\lambda}$ is an average, not $0$. Substituting $\lambda=1$ into (5.4) is invalid ($1-\lambda=0$). TD(1) is $\lambda\to 1$ of that average.
>
> Unrolling (1.10) $n$ times (the same pattern as (5.5)) gives
> ```math
> G_t=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n G_{t+n}.
> ```
> Section 5.2 already has
> ```math
> U_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n}).
> ```
> The first $n$ rewards match, so
> ```math
> U_t^{(n)}-G_t=\gamma^n\bigl(V(S_{t+n})-G_{t+n}\bigr).
> ```
> $G$ is bounded as in Chapter 1: with $\gamma\in[0,1)$ and bounded rewards, (1.8) converges. $V$ is a finite table, frozen while this backup is formed. With $\alpha\in(0,1]$ the update $V\leftarrow(1-\alpha)V+\alpha U$ is a convex combination; for one-step TD, $U=R+\gamma V(s')$, so if the table already lies in the same interval that bounds $G$, then $U$ stays in that interval and so does $V$. The factor $\gamma^n$ therefore sends the difference to $0$, hence $U_t^{(n)}\to G_t$ as $n\to\infty$. As $\lambda\to 1$, each fixed-$n$ weight vanishes, so the average in (5.4) is carried by large $n$, where $U_t^{(n)}$ is already near $G_t$. Thus $U_t^{\lambda}\to G_t$. The update becomes
> ```math
> V(S_t)\leftarrow V(S_t)+\alpha\bigl(G_t-V(S_t)\bigr),
> ```
> which is the Monte Carlo step (4.3).

Here $\lambda$ is not a power on $\gamma$ and not the horizon $n$: $n$-step TD (Section 5.2) picks one $U_t^{(n)}$, while $\lambda$ sets how fast the mix forgets longer backups. There is no TD(2) in the $\lambda$ naming. From here on, Section 5.1’s method is called TD(0) when contrasting with multi-step or $\lambda$ methods.

On a finite episode that ends at time $T$, the non-terminal states are $S_0,\ldots,S_{T-1}$. From a time $t$ in that episode, Section 5.2’s backup $U_t^{(n)}$ uses the next $n$ rewards and then $V(S_{t+n})$. Here $S_{t+n}$ is the state $n$ steps after $S_t$ on the same episode. That state is a non-terminal only while $t+n<T$. If $n\ge T{-}t$, the episode has already ended, so Section 5.2 gives $U_t^{(n)}=G_t$ (the lookup $V(\text{terminal})$ is $0$). The mix (5.4) is unchanged: every term with $n\ge T{-}t$ equals $G_t$. The weights still sum to $1$. The first $T{-}t{-}1$ of them add to $1-\lambda^{T-t-1}$, and the leftover $\lambda^{T-t-1}$ therefore sits on $G_t$:
```math
U_t^{\lambda}=\sum_{n=1}^{T-t-1}(1-\lambda)\lambda^{n-1}U_t^{(n)}+\lambda^{T-t-1}G_t.\quad (5.6)
```
Computing $U_t^{\lambda}$ needs the future rewards of the episode, so the forward-view algorithm below generates an episode first, then updates toward (5.6)—unlike online TD(0).

> [!NOTE]
>
> Input: policy $\pi$ to evaluate; $\lambda\in[0,1]$.
>
> Output: estimate $V\approx V^\pi$.
>
> 1.  Initialize $V(s)$ arbitrarily (e.g. $0$), with $V(\text{terminal})=0$.
>
> 2.  Loop for each episode:
>
>     - Generate an episode following $\pi$: $S_0,A_0,R_1,\;S_1,A_1,R_2,\;\ldots,\;S_{T-1},A_{T-1},R_T$.
>
>     - For each $t=0,1,\ldots,T{-}1$, compute $U_t^{\lambda}$ from (5.6) (same frozen table $V$ for every $t$).
>
>     - For each $t=0,1,\ldots,T{-}1$: $V(S_t)\leftarrow V(S_t)+\alpha\bigl(U_t^{\lambda}-V(S_t)\bigr)$.
>
> 3.  After enough episodes, return $V$ ($\approx V^\pi$).

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

def lambda_backups(episode, V, lam, gamma):
    """U_t^? for every t (backward recursion; same as (5.6))."""
    T = len(episode)
    U = [0.0] * T
    # last step: only R_T; bootstrap at terminal is 0
    U[T - 1] = episode[T - 1][2]
    for t in range(T - 2, -1, -1):
        r = episode[t][2]
        s_next = episode[t + 1][0]
        U[t] = r + gamma * ((1 - lam) * V[s_next] + lam * U[t + 1])
    return U

def td_lambda_prediction(pi, lam=0.5, alpha=0.1, gamma=0.99, max_episodes=500):
    # Initialize V(s) arbitrarily (algorithm step 1)
    V = {s: random.random() for s in STATES}
    n_ep = 0
    while True:  # for each episode
        episode = generate_episode(pi)
        U = lambda_backups(episode, V, lam, gamma)  # frozen V for every t
        for t in range(len(episode)):
            s = episode[t][0]
            V[s] += alpha * (U[t] - V[s])
        n_ep += 1
        if n_ep >= max_episodes:
            break
    return V

if __name__ == "__main__":
    pi = {s: random.choice(ACTIONS) for s in STATES}  # fixed policy
    print("TD(0):", td_lambda_prediction(pi, lam=0.0, alpha=0.1, gamma=0.9))
    print("TD(?):", td_lambda_prediction(pi, lam=0.5, alpha=0.1, gamma=0.9))
    print("TD(1):", td_lambda_prediction(pi, lam=1.0, alpha=0.1, gamma=0.9))
```

[← Previous: Monte Carlo Methods](04-monte-carlo-methods.md) | [Table of contents](../main.md#chapters) | [Next: TD Control: SARSA, Q-Learning, and Exploration](06-td-control.md) →
