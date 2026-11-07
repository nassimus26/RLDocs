# Chapter 9: Policy Gradients and REINFORCE

[← Previous: Deep Q-Networks and Double DQN](08-deep-q-networks-dqn-double-dqn.md) | [Table of contents](../main.md#chapters) | [Next: Actor-Critic Methods](10-actor-critic-methods.md) →


Chapters 5–11 mostly learn a policy **indirectly**. Policy iteration, SARSA, Q-learning, and DQN first estimate values $V$ or $Q$ (or $\hat{Q}$), then set $\pi$ by a greedy or $\varepsilon$-greedy rule on those values—for example $\pi(s)=\arg\max_a Q(s,a)$. That works when $\mathcal{A}$ is a small discrete set, as in the taxi’s few turns: score each action, pick among them.

When $\mathcal{A}$ is not like that, the same pattern breaks down. Suppose the agent must set a continuous control—how hard to brake, how far to turn the wheel. A value-based learner would need $\hat{Q}(s,a)$ for every real-valued $a$, then search for the best one at every step; that is not the discrete $\arg\max$ of Chapters 8–11. Or suppose the setting is rock–paper–scissors: any fixed pure choice (always rock, always paper, or always scissors) is beaten as soon as the opponent notices; the agent needs a mixed strategy $\pi(a\mid s)$. $\varepsilon$-greedy (6.7) does randomize, but non-greedy actions share $\varepsilon/|\mathcal{A}|$ equally, so it cannot learn a specific mix such as $70\%$–$30\%$, and if two actions have nearly equal $Q$ and a small update swaps which of the two is greedy, almost all probability $1-\varepsilon$ suddenly moves from one to the other. In both cases what should be learned is the action-selection rule itself—not a policy reconstructed after each value update.

**Policy gradient** methods learn $\pi$ **directly**. The policy is written $\pi_\theta(a\mid s)$, a distribution over actions with parameters $\theta$, so a continuous action law or a specific mix is represented by $\theta$ itself, without $\arg\max_a\hat{Q}$. The policy is scored by a single number $J(\theta)$, and $\theta$ is improved by gradient ascent on $J$; the name says exactly that, the gradient of the objective with respect to the policy’s parameters. The score is the expected episode return from the start state $s_0$:

```math
J(\theta)=V^{\pi_\theta}(s_0)=\mathbb{E}_{\pi_\theta}\bigl[G_0\mid S_0=s_0\bigr].\quad (9.1)
```

The score is taken at the start state $s_0$ because $G_0$ already sums every reward in the episode, so $V^{\pi_\theta}(s_0)$ scores the whole episode. Raising it requires better actions in every state visited from $s_0$, since those actions determine $G_0$.

Gradient ascent needs $\nabla_\theta J$, but $J$ is an expectation over episodes whose distribution depends on the unknown environment, so $\nabla_\theta J$ cannot be computed exactly. **REINFORCE** (), the basic policy-gradient method, estimates it from sampled episodes: run $\pi_\theta$ for a full episode, compute the return that followed each action, and change $\theta$ to make each action taken more likely, weighted by that return. An action followed by a high return becomes more probable in the state where it was taken; it is reinforced, which gives the method its name. Like the Monte Carlo methods of Chapter 4, REINFORCE learns from complete episodes and needs neither a model of the environment nor a value function.

## 9.1 Parameterized policies

Raising $J(\theta)$ requires a concrete $\pi_\theta$. From Chapter 1, $\pi(a\mid s)=P(A_t=a\mid S_t=s)$. A **parameterized policy** makes that probability depend on a parameter vector $\theta$:

```math
\pi_\theta(a\mid s)=P(A_t=a\mid S_t=s;\,\theta).
```

For each $s$, the numbers $\lbrace\pi_\theta(a\mid s)\rbrace_{a\in\mathcal{A}}$ must be nonnegative, sum to $1$, and change with $\theta$. That still needs a map from $(s,\theta)$ to a distribution over $\mathcal{A}$.

The softmax of Chapter 6 (6.9) is such a map: one real score per action becomes a distribution over $\mathcal{A}$. There the scores were action-values $Q(s,a)$. That coupling is the wrong object for policy gradients: $Q$ must track expected return and the Bellman backup, so its scale is tied to rewards and $\gamma$, and $\pi$ is only a byproduct of those values. What is needed instead is a parameterized state–action score whose only job is to set $\pi_\theta$—free to rise or fall as raising $J$ requires. Call that score a **preference** $h(s,a;\theta)$. Learning means updating $\theta$, which changes $h$, which changes $\pi_\theta$. Larger $h$ means larger probability after the softmax. Concretely, $\theta$ may be a table with one entry $\theta_{s,a}$ per pair ($h=\theta_{s,a}$), or the weights of a network that maps $s$ to $|\mathcal{A}|$ scores. For discrete $\mathcal{A}$, the parameterized policy is the softmax of these preferences:

```math
\pi_\theta(a\mid s)=\frac{\exp\bigl(h(s,a;\theta)\bigr)}{\sum_{b\in\mathcal{A}}\exp\bigl(h(s,b;\theta)\bigr)}.
```

For continuous $\mathcal{A}$ there is no finite list to softmax over, so $\pi_\theta(\cdot\mid s)$ is a continuous distribution. A common choice is Gaussian. Just as $\theta$ parameterizes preferences $h$ in the discrete case, here $\theta$ parameterizes a map from the state $s$ to the Gaussian’s mean $\mu_\theta(s)$ and standard deviation $\sigma_\theta(s)>0$. The density of that policy is

```math
\pi_\theta(a\mid s)=\frac{1}{\sqrt{2\pi}\,\sigma_\theta(s)}\exp\Biggl(-\frac{\bigl(a-\mu_\theta(s)\bigr)^2}{2\sigma_\theta(s)^2}\Biggr).
```

Actions are sampled from that distribution.

Either form—softmax of preferences or Gaussian—gives a concrete $\pi_\theta$ the agent can sample from. Raising $J(\theta)$ is then gradient ascent on $\theta$:

```math
\theta\leftarrow\theta+\alpha\,\nabla_\theta J(\theta).
```

The rest of this section derives a usable estimate of $\nabla_\theta J$ and turns it into the **REINFORCE** algorithm.

## 9.2 Trajectory form of $J(\theta)$

Raising $J$ by gradient ascent needs $\nabla_\theta J(\theta)$. By (9.1), $J(\theta)$ is the expected return of episodes that start at $s_0$ and are generated by $\pi_\theta$. A full episode is a trajectory
```math
\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\ldots,S_T).
```
Its probability under $\pi_\theta$ factors into the start-state distribution, the policy choices, and the environment transitions from Chapter 1:
```math
P(\tau;\theta)=P(S_0)\prod_{t=0}^{T-1}\pi_\theta(A_t\mid S_t)\,P(S_{t+1},R_{t+1}\mid S_t,A_t).\quad (9.2)
```
Only the $\pi_\theta$ factors depend on $\theta$. Write $R(\tau)$ for the return of $\tau$. That return is $G_0$ from Chapter 1, not the per-step reward $R_t$:

```math
R(\tau)=G_0=R_1+\gamma R_2+\gamma^2 R_3+\cdots.
```

The expectation in (9.1) averages $R(\tau)$ over trajectories according to those probabilities. In the discrete case each trajectory has probability mass $P(\tau;\theta)$, so

```math
J(\theta)=\sum_{\tau} P(\tau;\theta)\,R(\tau).
```

If trajectories form a continuous family, the sum becomes an integral with density $P(\tau;\theta)$:

```math
J(\theta)=\int P(\tau;\theta)\,R(\tau)\,d\tau.
```

The integral is definite: it runs over all trajectories. The two forms are the same idea; the steps below use the integral notation, and the discrete case replaces integrals by sums.

## 9.3 Policy gradient

The return $R(\tau)$ is produced by the environment along the trajectory; it does not depend on the policy parameters $\theta$. Only the distribution $P(\tau;\theta)$ does. Differentiate the integral form of $J$ accordingly—$\nabla_\theta$ hits $P$, not $R$:

```math
\nabla_\theta J(\theta)=\int \bigl(\nabla_\theta P(\tau;\theta)\bigr)\,R(\tau)\,d\tau.
```

For $u=P(\tau;\theta)>0$, $\nabla_\theta\log u=\frac{1}{u}\nabla_\theta u$, so $\nabla_\theta P(\tau;\theta)=P(\tau;\theta)\,\nabla_\theta\log P(\tau;\theta)$. Substitute:

```math
\nabla_\theta J(\theta)=\int P(\tau;\theta)\,\bigl(\nabla_\theta\log P(\tau;\theta)\bigr)\,R(\tau)\,d\tau.
```

The integrand has the form $P(\tau;\theta)\,f(\tau)$, where
```math
f(\tau)=\bigl(\nabla_\theta\log P(\tau;\theta)\bigr)\,R(\tau)
```
is a function of the trajectory. By definition of expectation under the density $P(\tau;\theta)$,

```math
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Bigl[\bigl(\nabla_\theta\log P(\tau;\theta)\bigr)\,R(\tau)\Bigr].\quad (9.3)
```

The subscript $\tau\sim\pi_\theta$ means the expectation is over trajectories generated by acting with $\pi_\theta$ (together with the environment): sample $\tau$ from $P(\tau;\theta)$. It remains to evaluate $\nabla_\theta\log P(\tau;\theta)$ from (9.2). The log of the product is a sum:

```math
\begin{aligned}\log P(\tau;\theta)&=\log P(S_0)+\sum_{t=0}^{T-1}\log\pi_\theta(A_t\mid S_t)\\&\qquad+\sum_{t=0}^{T-1}\log P(S_{t+1},R_{t+1}\mid S_t,A_t).\end{aligned}\quad (9.4)
```

Differentiate with respect to $\theta$. The terms $P(S_0)$ and $P(s',r\mid s,a)$ do not depend on $\theta$, so

```math
\nabla_\theta\log P(\tau;\theta)=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(A_t\mid S_t).\quad (9.5)
```

Substitute (9.5) into (9.3):

```math
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Biggl[\Biggl(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(A_t\mid S_t)\Biggr)R(\tau)\Biggr].\quad (9.6)
```

The gradient uses only $\pi_\theta$ and returns from sampled episodes.

## 9.4 REINFORCE

The factor $\nabla_\theta\log\pi_\theta(A_t\mid S_t)$ in (9.6) is a function of $(S_t,A_t)$ and $\theta$. Write
```math
\psi_\theta(s,a)=\nabla_\theta\log\pi_\theta(a\mid s),
```
so (9.6) with $R(\tau)=G_0$ is
```math
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Biggl[\sum_{t=0}^{T-1}G_0\,\psi_\theta(S_t,A_t)\Biggr].
```
The return $G_0$ multiplies every $\psi_\theta(S_t,A_t)$, and it includes rewards received before $A_t$ was taken. For each $t$, split $G_0$ into the rewards before time $t$ and the rest:
```math
G_0=\sum_{k=0}^{t-1}\gamma^{k}R_{k+1}+\gamma^{t}G_t,\qquad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1},
```
where $G_t$ is the return-to-go from time $t$. Substituting the split and applying linearity (A.4) separates the gradient into a past part and a future part:
```math
\nabla_\theta J(\theta)=\underbrace{\sum_{t=0}^{T-1}\sum_{k=0}^{t-1}\gamma^{k}\,\mathbb{E}\bigl[R_{k+1}\,\psi_\theta(S_t,A_t)\bigr]}_{\text{past}}+\underbrace{\mathbb{E}_{\tau\sim\pi_\theta}\Biggl[\sum_{t=0}^{T-1}\gamma^{t}G_t\,\psi_\theta(S_t,A_t)\Biggr]}_{\text{future}}.
```
Every term of the past part is zero, by <a href="#eq:E.2" data-reference-type="eqref" data-reference="eq:E.2">[eq:E.2]</a> in . The reward $R_{k+1}$ with $k<t$ was received before $A_t$ was chosen, and the policy draws $A_t$ from $\pi_\theta(\cdot\mid S_t)$ whatever those earlier rewards were, so
```math
\mathbb{E}\bigl[R_{k+1}\,\psi_\theta(S_t,A_t)\bigr]=0\qquad\text{for } k<t.
```
Only the future part remains. Replacing $\psi_\theta(S_t,A_t)$ by its definition $\nabla_\theta\log\pi_\theta(A_t\mid S_t)$,
```math
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Biggl[\sum_{t=0}^{T-1}\gamma^{t}G_t\,\nabla_\theta\log\pi_\theta(A_t\mid S_t)\Biggr].\quad (9.7)
```
In (9.7), each action is weighted only by the return that followed it, and every quantity can be read from a sampled episode. **REINFORCE** estimates this gradient from sampled episodes.

The same gradient can be written with action-values. By definition $Q^{\pi_\theta}(s,a)=\mathbb{E}[G_t\mid S_t=s,A_t=a]$, so $\mathbb{E}[G_t\mid S_t,A_t]=Q^{\pi_\theta}(S_t,A_t)$. Starting from (9.7),
```math
\begin{aligned}\nabla_\theta J(\theta)&=\mathbb{E}\Biggl[\sum_{t=0}^{T-1}\gamma^{t}G_t\,\psi_\theta(S_t,A_t)\Biggr]\\&=\sum_{t=0}^{T-1}\gamma^{t}\,\mathbb{E}\bigl[G_t\,\psi_\theta(S_t,A_t)\bigr]\quad\text{[A.4]}\\&=\sum_{t=0}^{T-1}\gamma^{t}\,\mathbb{E}\Bigl[\mathbb{E}\bigl[G_t\,\psi_\theta(S_t,A_t)\bigm|S_t,A_t\bigr]\Bigr]\quad\text{[A.8]}\\&=\sum_{t=0}^{T-1}\gamma^{t}\,\mathbb{E}\Bigl[\psi_\theta(S_t,A_t)\,\mathbb{E}\bigl[G_t\bigm|S_t,A_t\bigr]\Bigr]\quad\text{[A.5; }\psi_\theta(S_t,A_t)\text{ is fixed given }S_t,A_t\text{]}\\&=\sum_{t=0}^{T-1}\gamma^{t}\,\mathbb{E}\bigl[\psi_\theta(S_t,A_t)\,Q^{\pi_\theta}(S_t,A_t)\bigr]\quad\text{[definition of }Q^{\pi_\theta}\text{]}\\&=\mathbb{E}\Biggl[\sum_{t=0}^{T-1}\gamma^{t}Q^{\pi_\theta}(S_t,A_t)\,\psi_\theta(S_t,A_t)\Biggr].\quad\text{[A.4]}\end{aligned}
```
Replacing $\psi_\theta(S_t,A_t)$ by its definition $\nabla_\theta\log\pi_\theta(A_t\mid S_t)$,
```math
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Biggl[\sum_{t=0}^{T-1}\gamma^{t}Q^{\pi_\theta}(S_t,A_t)\,\nabla_\theta\log\pi_\theta(A_t\mid S_t)\Biggr].\quad (9.7a)
```
Form (9.7a) needs $Q^{\pi_\theta}$, which is unknown. In (9.7), $G_t$ is a single sample whose conditional mean is $Q^{\pi_\theta}(S_t,A_t)$.

The same gradient can also be written without an explicit sum over time: average $Q^{\pi_\theta}(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)$ under the discounted state–action visits of $\pi_\theta$ from $s_0$. That identity is the **policy gradient theorem** (); derives it from the Bellman equation for $V^{\pi_\theta}$ and shows that it equals (9.7a).

The expectation in (9.7) cannot be computed, but it can be approximated by an average over episodes $\tau^{(1)},\ldots,\tau^{(N)}$ generated by $\pi_\theta$, as in the Monte Carlo methods of Chapter 4:
```math
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T_i-1}\gamma^{t}G_t^{(i)}\,\nabla_\theta\log\pi_\theta\bigl(A_t^{(i)}\mid S_t^{(i)}\bigr),
```
where the superscript $(i)$ marks quantities from episode $i$. REINFORCE updates $\theta$ after every single episode, so $N=1$: the average over episodes reduces to the one episode’s sum over $t$, and the superscript is dropped. Generate that episode, compute each $G_t$, and put the estimate in place of $\nabla_\theta J$ in the gradient-ascent step $\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)$:
```math
\theta\leftarrow\theta+\alpha\sum_{t=0}^{T-1}\gamma^{t}G_t\,\nabla_\theta\log\pi_\theta(A_t\mid S_t).\quad (9.8)
```

Comparing (9.8) with the gradient-descent step $\theta\leftarrow\theta-\alpha\nabla_\theta L(\theta)$ gives the loss, since each $G_t$ is a fixed number computed from the episode:
```math
L(\theta)=-\sum_{t=0}^{T-1}\gamma^{t}G_t\,\log\pi_\theta(A_t\mid S_t).\quad (9.9)
```

> [!NOTE]
>
> Input: differentiable policy $\pi_\theta(a\mid s)$; step size $\alpha$; discount $\gamma$; number of episodes $M$.
>
> Output: trained policy $\pi_\theta$.
>
> 1.  Initialize $\theta$ (e.g. randomly).
>
> 2.  Loop for episode $=1,2,\ldots,M$:
>
>     - Generate an episode $\tau=(S_0,A_0,R_1,\ldots,S_T)$ by sampling $A_t\sim\pi_\theta(\cdot\mid S_t)$ until termination.
>
>     - For each $t=0,1,\ldots,T-1$:
>
>       - $G_t\leftarrow\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}$.
>
>     - $\theta\leftarrow\theta+\alpha\sum_{t=0}^{T-1}\gamma^{t}G_t\,\nabla_\theta\log\pi_\theta(A_t\mid S_t)$.
>
> 3.  Return $\pi_\theta$.

```python
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical

STATES = ["A", "B", "C", "D", "E", "F", "G", "H"]  # non-terminal states
TERMINALS = {"I", "J"}  # more than one terminal is fine
ACTIONS = list(range(4))  # action set A = {0,1,2,3}

def step(s, a):
    """One transition: next state uniform over non-terminals and terminals."""
    s_next = random.choice(STATES + list(TERMINALS))
    r = 1.0 if s_next in TERMINALS else 0.0
    return s_next, r

def x_s(s):
    """One-hot vector for state s (network input)."""
    x = torch.zeros(len(STATES))
    x[STATES.index(s)] = 1.0
    return x

def make_h():
    """Preference network h(s,*;theta): one logit per action. Its weights are theta."""
    return nn.Sequential(
        nn.Linear(len(STATES), 64),
        nn.ReLU(),
        nn.Linear(64, len(ACTIONS)),
    )

def pi_theta(h, s):
    """Returns pi_theta(*|s) = softmax(h(s,*;theta)), a distribution over actions."""
    return Categorical(logits=h(x_s(s)))

def act(h, s):
    """Agent's action: A ? pi_theta(*|s)."""
    return int(pi_theta(h, s).sample().item())

def generate_episode(h):
    """Roll out one episode under pi_theta: lists of log pi_theta(A_t|S_t) and R_{t+1}."""
    logps, rewards = [], []
    s = "A"  # S_0
    while s not in TERMINALS:
        dist = pi_theta(h, s)
        a = dist.sample()  # A_t ? pi_theta(*|S_t)
        s_next, r = step(s, int(a.item()))
        logps.append(dist.log_prob(a))
        rewards.append(r)
        s = s_next
    return logps, rewards

def reinforce(alpha=0.01, gamma=0.99, max_episodes=500):
    # 1. Initialize theta randomly (weights of h)
    h = make_h()
    opt = torch.optim.SGD(h.parameters(), lr=alpha)  # updates theta with step size ?
    n_ep = 0
    while True:  # 2. Loop for each episode
        logps, rewards = generate_episode(h)
        T = len(rewards)

        # For each t: G_t <- sum_{k=t}^{T-1} gamma^{k-t} R_{k+1}  (rewards[k] is R_{k+1})
        G = [sum(gamma ** (k - t) * rewards[k] for k in range(t, T)) for t in range(T)]

        # theta <- theta + ? sum_t gamma^t G_t ?_theta log pi_theta(A_t|S_t): one SGD step on L(theta) (9.9)
        loss = -sum(gamma**t * G[t] * logps[t] for t in range(T))
        opt.zero_grad()
    loss.backward()
        opt.step()

        n_ep += 1
        if n_ep >= max_episodes:
            break
    return h  # 3. Return pi_theta (h holds the trained theta)

if __name__ == "__main__":
    h = reinforce(alpha=0.01, gamma=0.9)
    print("pi_theta(*|A):", [round(p, 3) for p in pi_theta(h, "A").probs.tolist()])
    print("action at A:", act(h, "A"))
```

REINFORCE is on-policy and Monte Carlo: each update uses full returns from episodes generated by the current $\pi_\theta$. The estimator (9.7) is unbiased but can have high variance because $G_t$ sums many noisy rewards. Chapter 10 improves that weight the way Chapters 6–7 improved value targets: first center $G_t$ with a baseline, then replace it by a TD error.

[← Previous: Deep Q-Networks and Double DQN](08-deep-q-networks-dqn-double-dqn.md) | [Table of contents](../main.md#chapters) | [Next: Actor-Critic Methods](10-actor-critic-methods.md) →
