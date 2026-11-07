[← Previous: Bellman Expectation Derivation](b-bellman-expectation-derivation.md) | [Table of contents](../main.md#chapters) | [Next: Robbins--Monro Step-Size Conditions](d-robbins-monro-step-size-conditions.md) →

# Appendix C: Bellman Optimality Derivation

The goal is to derive the Bellman optimality equations (C.2) and (C.5).

## C.1 Definitions

```math
V^*(s) = \max_\pi V^\pi(s),\qquad Q^*(s,a) = \max_\pi Q^\pi(s,a).
```

Equivalently,

```math
V^*(s)=\max_\pi\mathbb{E}_\pi[G_t \mid S_t=s],\qquad Q^*(s,a)=\max_\pi\mathbb{E}_\pi[G_t \mid S_t=s, A_t=a].
```

We expand $V^*$ first, then $Q^*$.

## C.2 State-value $V^*$

From (C.1), $V^*(s)=\max_\pi V^\pi(s)$. From (B.1),

```math
V^\pi(s)=\sum_a \pi(a \mid s)\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^\pi(s') \right].
```

Therefore

```math
V^*(s)=\max_\pi\sum_a \pi(a \mid s)\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^\pi(s') \right].
```

$V^*$ means the agent acts optimally at every state. After one step it is at $s'$, so the remaining value is $V^*(s')$. (Markov: the past action does not change that.) Substitute $V^*(s')$ for $V^\pi(s')$:

```math
V^*(s)=\max_\pi\sum_a \pi(a \mid s)\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].
```

For every action $a$,

```math
\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right]\le\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].
```

Therefore

```math
\begin{aligned}V^*(s)&=\max_\pi\sum_a \pi(a \mid s)\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right] \\&\le\max_\pi\sum_a \pi(a \mid s)\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].\end{aligned}
```

In the inner $\max_a$, the action is already fixed to the one that maximizes $\sum_{s',r}P(s',r\mid s,a)[r+\gamma V^*(s')]$. That choice does not change as the outer sum runs over the possible values of $a$ under $\pi$, so the inner max can be taken outside the outer sum. Then use $\sum_a \pi(a \mid s)=1$:

```math
\begin{aligned}V^*(s)&\le\max_\pi\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right]\sum_a \pi(a \mid s) \\&=\max_\pi\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right] \\&=\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right]\quad\text{[inner sum does not depend on }\pi\text{]} .\end{aligned}
```

For the other direction: by definition $V^*(s)=\max_\pi(\cdots)$, so for any other policy $\hat{\pi}$ --- whether optimal or not ---

```math
V^*(s)\ge\sum_a \hat{\pi}(a \mid s)\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].
```

Choose $\hat{\pi}$ that puts probability $1$ on one action $\hat{a}$ attaining the max, and $0$ on every other action. Then

```math
\begin{aligned}\sum_a \hat{\pi}(a \mid s)\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right]&=\sum_{s',r}P(s',r \mid s,\hat{a}) \left[ r + \gamma V^*(s') \right] \\&=\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right],\end{aligned}
```

and therefore

```math
V^*(s)\ge\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].
```

The two directions give (C.2):

```math
V^*(s)=\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].
```

## C.3 Action-value $Q^*$

From (C.1), $Q^*(s,a)=\max_\pi Q^\pi(s,a)$. From (B.17),

```math
Q^\pi(s,a)=\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^\pi(s') \right].
```

Therefore

```math
Q^*(s,a)=\max_\pi\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^\pi(s') \right].
```

The sum over $P$ does not depend on $\pi$. Only the continuation after $s'$ does. As in deriving (C.2), acting optimally from $s'$ onward means $V^\pi(s')$ becomes $V^*(s')$:

```math
Q^*(s,a)=\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].
```

Compare with (C.2): the right-hand side of (C.2) is $\max_a$ of this same sum. Therefore

```math
V^*(s) = \max_a Q^*(s,a).
```

At the next state this is $V^*(s')=\max_{a'} Q^*(s',a')$ by (C.4). Substitute into (C.3):

```math
Q^*(s,a)=\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma \max_{a'} Q^*(s',a') \right].
```

## C.4 Optimal policy

We want a policy $\pi^*$ that attains the optimum. (C.2) and (C.5) already give that optimum as a maximum over actions, so $\pi^*$ is obtained by choosing actions that attain those maxima.

From (C.2), an optimal policy at $s$ chooses an action attaining this max:

```math
\pi^*(s)=\arg\max_a\sum_{s',r}P(s',r \mid s,a) \left[ r + \gamma V^*(s') \right].
```

From (C.4), the same policy is

```math
\pi^*(s) = \arg\max_a Q^*(s,a).
```

(Acting from $V^*$ needs $P$; acting from the table $Q^*$ does not. Computing either table from (C.2) or (C.5) uses $P$.)

- **Greedy.** Put probability $1$ on one maximizing action. If only one action maximizes, this is the only optimal choice at $s$. If several do, any one of them may be chosen with probability $1$.
- **Randomized among maximizers.** If several actions maximize, $\pi^*$ may put any probabilities on those actions only. All other actions get probability $0$. This is still optimal: every action that can be chosen still attains the $\max$.


[← Previous: Bellman Expectation Derivation](b-bellman-expectation-derivation.md) | [Table of contents](../main.md#chapters) | [Next: Robbins--Monro Step-Size Conditions](d-robbins-monro-step-size-conditions.md) →
