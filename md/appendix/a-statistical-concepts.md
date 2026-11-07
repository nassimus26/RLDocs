[← Previous: Appendix Index](../appendix.md) | [Table of contents](../main.md#chapters) | [Next: Bellman Expectation Derivation](b-bellman-expectation-derivation.md) →

# Appendix A: Statistical Concepts

The following tools are used in the Bellman derivations that follow, in the Robbins--Monro argument of Appendix D, and later in off-policy Monte Carlo. Probability relations first, then expectation, then variance, then the law of large numbers, then Monte Carlo estimation, then importance sampling, then the Markov property.

## A.1 Probability

### A.1.1 Law of total probability (LOTP)

If $B_1,\ldots,B_n$ are mutually exclusive and exhaustive,

```math
P(A)=\sum_i P(A\mid B_i)\, P(B_i)=\sum_i P(A, B_i).
```

## A.2 Expectation

### A.2.1 Definition

For a discrete random variable $X$,

```math
\mathbb{E}[X]=\sum_x x\, P(X=x).
```

### A.2.2 Conditional expectation

Conditioning on $Y=y$ (or on an event),

```math
\mathbb{E}[X\mid Y=y]=\sum_x x\, P(X=x\mid Y=y).
```

### A.2.3 Linearity of (conditional) expectation

For random variables $Y_1,\ldots,Y_n$ and constants $a_1,\ldots,a_n$,

```math
\mathbb{E}[a_1 Y_1+\cdots+a_n Y_n]=a_1\mathbb{E}[Y_1]+\cdots+a_n\mathbb{E}[Y_n],
```

and conditioning on a random variable $X$ (or on an event):

```math
\mathbb{E}[a_1 Y_1+\cdots+a_n Y_n\mid X]=a_1\mathbb{E}[Y_1\mid X]+\cdots+a_n\mathbb{E}[Y_n\mid X].
```

**Worked expansion.** For discrete $X_1,X_2$, constants $a,b$, and $Y=y$:

```math
\begin{aligned}\mathbb{E}[a X_1+b X_2\mid Y=y]&=a\,\mathbb{E}[X_1\mid Y=y]+b\,\mathbb{E}[X_2\mid Y=y]\quad\text{[A.5; linearity]}\\&=a\sum_{x_1} x_1\, P(X_1=x_1\mid Y=y)+b\sum_{x_2} x_2\, P(X_2=x_2\mid Y=y)\quad\text{[A.3; conditional expectation]}\\&=a\sum_{x_1} x_1\sum_{x_2} P(X_1=x_1,X_2=x_2\mid Y=y)+b\sum_{x_2} x_2\sum_{x_1} P(X_1=x_1,X_2=x_2\mid Y=y)\quad\text{[A.1; LOTP]}\end{aligned}
```

```math
\mathbb{E}[a X_1+b X_2\mid Y=y]=\sum_{x_1,x_2} (a x_1+b x_2)\, P(X_1=x_1,X_2=x_2\mid Y=y).
```

(In this book one often takes $a=1$, $b=\gamma$, with $X_1,X_2$ a reward and a return.)

### A.2.4 Expanding a conditional expectation over a partition

Start from the definition, then apply LOTP to the conditional probability over the values of a discrete random variable $Y$:

```math
\begin{aligned}\mathbb{E}[X\mid A]&=\sum_x x\, P(X=x\mid A)\quad\text{[A.3; conditional expectation]}\\&=\sum_x x\sum_y P(X=x\mid A,Y=y)\, P(Y=y\mid A)\quad\text{[A.1; LOTP]}\\&=\sum_y\Bigl(\sum_x x\, P(X=x\mid A,Y=y)\Bigr)P(Y=y\mid A)\end{aligned}
```

```math
\mathbb{E}[X\mid A]=\sum_y \mathbb{E}[X\mid A,Y=y]\, P(Y=y\mid A).
```

### A.2.5 Law of total expectation (tower property)

Start from the definition of $\mathbb{E}[X]$, expand $P(X=x)$ by LOTP over the values of a discrete $Y$, then regroup:

```math
\begin{aligned}\mathbb{E}[X]&=\sum_x x\, P(X=x)\quad\text{[A.2]}\\&=\sum_x x\sum_y P(X=x\mid Y=y)\, P(Y=y)\quad\text{[A.1; LOTP]}\\&=\sum_y\Bigl(\sum_x x\, P(X=x\mid Y=y)\Bigr)P(Y=y)\\&=\sum_y \mathbb{E}[X\mid Y=y]\, P(Y=y)\quad\text{[A.3]}\\&=\mathbb{E}\bigl[\mathbb{E}[X\mid Y]\bigr].\end{aligned}
```

So

```math
\mathbb{E}[X]=\mathbb{E}\bigl[\mathbb{E}[X\mid Y]\bigr].
```

The same argument with the event $\lbrace Z=z\rbrace$ in place of the unconditional law gives the conditional tower. Apply (A.7) with $A=\lbrace Z=z\rbrace$ and partition by $Y$:

```math
\begin{aligned}\mathbb{E}[X\mid Z=z]&=\sum_y \mathbb{E}[X\mid Z=z,Y=y]\, P(Y=y\mid Z=z)\quad\text{[A.7]}\\&=\sum_y \mathbb{E}[X\mid Y=y,Z=z]\, P(Y=y\mid Z=z).\end{aligned}
```

For each fixed pair $(y,z)$, the number $\mathbb{E}[X\mid Y=y,Z=z]$ depends only on those values. Collecting them into a random variable that is a **function of $(Y,Z)$**,

```math
g(Y,Z):=\mathbb{E}[X\mid Y,Z],
```

the sum above is the conditional expectation of that function given $Z=z$:

```math
\begin{aligned}\mathbb{E}[X\mid Z=z]&=\sum_y g(y,z)\, P(Y=y\mid Z=z)\\&=\mathbb{E}\bigl[g(Y,Z)\bigm|\ Z=z\bigr]\\&=\mathbb{E}\bigl[\mathbb{E}[X\mid Y,Z]\bigm|\ Z=z\bigr].\end{aligned}
```

Writing the same identity for every $z$ as a random-variable equation:

```math
\mathbb{E}[X\mid Z]=\mathbb{E}\bigl[\mathbb{E}[X\mid Y,Z]\bigm|\ Z\bigr].
```

In discrete form with $Y$ taking values $y$ (the expansion used above):

```math
\mathbb{E}[X\mid Z=z]=\sum_y \mathbb{E}[X\mid Y=y,Z=z]\, P(Y=y\mid Z=z).
```

So (A.10) is (A.9) written pointwise at $Z=z$; (A.8) is the special case of (A.9) with no conditioning on $Z$ (or with $Z$ trivial).

### A.2.6 Expectation of a function of a discrete random variable (Law of the Unconscious Statistician)

The identity $\mathbb{E}[f(X)]=\sum_x f(x)\,P(X=x)$ is also called the **law of the unconscious statistician (LOTUS)**, or sometimes the **law of the lazy statistician**: one need not first find the distribution of $Y=f(X)$; the expectation follows from the distribution of $X$ alone.

Let $X$ be discrete and set $Y=f(X)$. Then $P(Y=y)$ is the sum of all probabilities of those $x$ for which $f(x)=y$:

```math
P(Y=y)=\sum_{x:\,f(x)=y} P(X=x).\quad\text{[A.1; LOTP]}
```

Then

```math
\begin{aligned}\mathbb{E}[f(X)]&=\mathbb{E}[Y]=\sum_y y\, P(Y=y)\\&=\sum_y y(\sum_{x:\,f(x)=y} P(X=x))\\&=\sum_y(\sum_{x:\,f(x)=y} y\, P(X=x))\\&=\sum_y(\sum_{x:\,f(x)=y} f(x)\, P(X=x))\quad\text{[}y=f(x)\text{ on each inner sum]}\\&=\sum_x f(x)\, P(X=x).\end{aligned}
```

The last equality collapses the nested sums. For each fixed $y$, the inner sum runs only over the group $\lbrace x:f(x)=y\rbrace$: any $x$ with $f(x)\neq y$ is missing from that group. Because $f$ is a function, every $x$ maps to exactly one $y$, so each $x$ appears in **exactly one** group, and every $x$ appears in **some** group. Summing $f(x)\,P(X=x)$ group by group is therefore the same as summing once over all $x$.

**Example.** Suppose the possible values of $X$ are $\lbrace1,2,3\rbrace$ and $f(1)=f(2)=a$, $f(3)=b$. The double sum is two groups:

```math
\sum_{x:\,f(x)=a} f(x)\,P(X=x)+\sum_{x:\,f(x)=b} f(x)\,P(X=x)=\bigl(f(1)P(X=1)+f(2)P(X=2)\bigr)+f(3)P(X=3),
```

which is exactly $\sum_{x\in\lbrace1,2,3\rbrace} f(x)\,P(X=x)$. In the $a$-group, $x=3$ is missing; in the $b$-group, $x=1$ and $x=2$ are missing; no $x$ appears in both.

(The same steps with $P(\,\cdot\mid A)$ give $\mathbb{E}[f(X)\mid A]=\sum_x f(x)\,P(X=x\mid A)$.)

### A.2.7 Variance and standard deviation

Write $\mu=\mathbb{E}[X]$. The **variance** is the expected squared deviation from that mean,

```math
\mathrm{Var}(X)=\mathbb{E}\bigl[(X-\mu)^2\bigr].
```

The **standard deviation** is the positive square root,

```math
\sigma_X=\sqrt{\mathrm{Var}(X)}.
```

So $\mathrm{Var}(X)=\sigma_X^2$. Expanding the square in (A.17) and using linearity (A.5) with $\mathbb{E}[X]=\mu$ gives the computational form

```math
\mathrm{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2.
```

If $\mathbb{E}[X]=0$, then (A.19) is $\mathrm{Var}(X)=\mathbb{E}[X^2]$. For a finite list $x_1,\ldots,x_n$ with list mean $\bar x$, the same square-root of mean squared deviation from $\bar x$ is written $\mathrm{std}$.

## A.3 Law of large numbers

The **weak law of large numbers** says: if $x_1,\ldots,x_n$ are i.i.d.\ samples from the distribution of a random variable $X$ with expectation $\mu=\mathbb{E}[X]$ and finite variance $\mathrm{Var}(X)=\sigma_X^2$, as in (A.17)--(A.18), then the sample mean of those draws has expectation $\mu$ and converges to $\mu$. Write the sample mean as

```math
\bar{x}_n=\frac{1}{n}\sum_{i=1}^{n} x_i.
```

Then $\mathbb{E}[\bar{x}_n]=\mu$, and $\bar{x}_n\to\mu$ in probability as $n\to\infty$.

**Proof.**

Linearity (A.4) gives

```math
\mathbb{E}[\bar{x}_n]=\frac{1}{n}\sum_{i=1}^{n} \mathbb{E}[x_i]=\mu.
```

So $\bar{x}_n$ is unbiased for $\mu$.

For a constant $a$, (A.17) gives $\mathrm{Var}(aX)=a^2\mathrm{Var}(X)$. For independent $X$ and $Y$,

```math
\begin{aligned}\mathrm{Var}(X+Y)&=\mathbb{E}\bigl[((X-\mu_X)+(Y-\mu_Y))^2\bigr]\\&=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathbb{E}[(X-\mu_X)(Y-\mu_Y)].\end{aligned}
```

Independence factors the cross term:

```math
\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]=\mathbb{E}[X-\mu_X]\,\mathbb{E}[Y-\mu_Y].
```

Each factor is the expectation of a deviation from the mean, so

```math
\mathbb{E}[X-\mu_X]=\mathbb{E}[X]-\mu_X=\mu_X-\mu_X=0,
```

and likewise $\mathbb{E}[Y-\mu_Y]=\mathbb{E}[Y]-\mu_Y=\mu_Y-\mu_Y=0$. The product is therefore $0$, and

```math
\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y).
```

Repeating for $n$ i.i.d.\ samples,

```math
\mathrm{Var}(\bar{x}_n)=\mathrm{Var}\!(\frac{1}{n}\sum_{i=1}^{n} x_i)=\frac{1}{n^2}\sum_{i=1}^{n} \mathrm{Var}(x_i)=\frac{\sigma_X^2}{n}.
```

The standard deviation of $\bar{x}_n$ is $\sigma_X/\sqrt{n}$. As $n\to\infty$, that variance goes to $0$. Chebyshev's inequality bounds how far a random variable can sit from its mean: for a random variable $Y$ with mean $\nu$ and finite variance, and every $\varepsilon>0$,

```math
P\bigl(|Y-\nu|\ge\varepsilon\bigr)\le\frac{\mathrm{Var}(Y)}{\varepsilon^2}.
```

Taking $Y=\bar{x}_n$ and $\nu=\mu$ with (A.22) gives

```math
P\bigl(|\bar{x}_n-\mu|\ge\varepsilon\bigr)\le\frac{\mathrm{Var}(\bar{x}_n)}{\varepsilon^2}=\frac{\sigma_X^2}{n\varepsilon^2}\to 0.
```

So $\bar{x}_n\to\mu$ in probability, which is the claim at the start of this section. The same Chebyshev bound is used in Appendix D.

## A.4 Monte Carlo estimation

**Monte Carlo** estimates an expectation $\mathbb{E}[f(X)]$ by drawing i.i.d.\ samples $x_1,\ldots,x_n$ from the distribution of $X$ and averaging the observed values of $f$. The estimator is

```math
\bar{f}_n=\frac{1}{n}\sum_{i=1}^{n} f(x_i).
```

If $\mathrm{Var}(f(X))$ is finite, then $\mathbb{E}[\bar{f}_n]=\mathbb{E}[f(X)]$, and $\bar{f}_n\to\mathbb{E}[f(X)]$ in probability as $n\to\infty$. Chapter 4 uses it with $f$ the return $G_t$ from a visit to $s$.

**Proof.**
Replace each $x_i$ in (A.20) by $f(x_i)$, which is legitimate by (A.12). Then (A.21)--(A.23) apply with $X$ replaced by $f(X)$, so $\bar{f}_n$ is unbiased for $\mathbb{E}[f(X)]$ and converges to it.

## A.5 Importance sampling

Importance sampling estimates an expectation under a target $p$ from samples drawn under a proposal $q$, by reweighting each sample with $w=p/q$. Example: estimate mean height in the population ($p$) from players at a basketball camp ($q$); tall players are common at camp but rare in the population, so $w$ downweights them.

Expectation is still (A.2): $\mathbb{E}[Y]=\sum_y y\,P(Y=y)$. When $Y=f(X)$ and $X$ has distribution $p$, (A.12) writes that as a sum over $x$: 

```math
\mathbb{E}_{x\sim p}[f(x)]=\sum_x f(x)\, p(x)
```

(for discrete $x$; continuous cases replace the sum by an integral). If we could draw $x_1,\ldots,x_N$ from $p$, the Monte Carlo average (A.24) of the values $f(x_i)$ would estimate that expectation.

Often we **cannot** (or do not) sample from $p$. Instead we have samples from a different distribution $q$ (the **proposal** or **behavior** distribution), with $q(x)>0$ whenever $p(x)>0$ and $f(x)\neq 0$. A plain average under $q$ estimates $\mathbb{E}_{x\sim q}[f(x)]$, which is the wrong target. Importance sampling rewrites the $p$-expectation as a $q$-expectation by inserting the ratio $p/q$: 

```math
\begin{aligned}\mathbb{E}_{x\sim p}[f(x)]&=\sum_x f(x)\, p(x)\\&=\sum_x f(x)\, \frac{p(x)}{q(x)}\, q(x)\\&=\sum_x w(x)\, f(x)\, q(x)\\&=\mathbb{E}_{x\sim q}\bigl[w(x)\, f(x)\bigr],\qquad w(x)=\frac{p(x)}{q(x)}.\end{aligned}
```

The weight has mean one under $q$:

```math
\begin{aligned}\mathbb{E}_{x\sim q}[w(x)]&=\sum_x w(x)\, q(x)\\&=\sum_x \frac{p(x)}{q(x)}\, q(x)\\&=\sum_x p(x)\\&=1.\end{aligned}
```

Dividing the right-hand side of (A.13) by this identity,

```math
\begin{aligned}\mathbb{E}_{x\sim p}[f(x)]&=\mathbb{E}_{x\sim q}\bigl[w(x)\, f(x)\bigr]\\&=\frac{\mathbb{E}_{x\sim q}\bigl[w(x)\, f(x)\bigr]}{\mathbb{E}_{x\sim q}[w(x)]}.\end{aligned}
```

With i.i.d. samples $x_i\sim q$, estimate the expectations by sample means. Estimating only the numerator of (A.14) (equivalently, the last line of (A.13)) gives **ordinary** importance sampling:

```math
\mathbb{E}_{x\sim p}[f(x)]\approx\frac{1}{N}\sum_{i=1}^{N} w(x_i)\, f(x_i).
```

Estimating both the numerator and the denominator of (A.14) gives **weighted** (self-normalized) importance sampling:

```math
\mathbb{E}_{x\sim p}[f(x)]\approx\frac{\tfrac{1}{N}\sum_{i=1}^{N} w(x_i)\, f(x_i)}{\tfrac{1}{N}\sum_{i=1}^{N} w(x_i)}=\frac{\sum_{i=1}^{N} w(x_i)\, f(x_i)}{\sum_{i=1}^{N} w(x_i)}.
```

A single term $w(x)\,f(x)$ is already an **unbiased** estimator of $\mathbb{E}_p[f]$, since its mean equals $\mathbb{E}_p[f]$ by (A.13). Ordinary importance sampling (A.15) averages $N$ such terms, so it is unbiased for $\mathbb{E}_p[f]$ as well. Weighted importance sampling (A.16) estimates the same target but is **biased** for finite $N$. It divides two random averages built from the same samples, and the mean of a ratio need not equal the ratio of the means. That bias goes away as $N$ grows. The cost of ordinary IS shows up in variance. If a few samples carry large weights and many carry small ones, (A.15) is driven by those few terms and can jump a lot from one run to the next. Weighted (A.16) is usually stabler. A large $w$ increases both the numerator and the denominator, so no single sample can take more than its share of the total weight.

If $q$ oversamples a region relative to $p$, then $w<1$ there (downweight); if $q$ undersamples it, then $w>1$ (upweight). Importance sampling **reweights** every draw. Rejection sampling instead **accepts or drops**; that procedure, and the hard-filter case, is Appendix H.

## A.6 Markov property

Given $S_{t+1}=s'$, the remaining return $G_{t+1}$ does not depend on $(S_t,A_t)$:

```math
\mathbb{E}_\pi[G_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s']=\mathbb{E}_\pi[G_{t+1}\mid S_{t+1}=s'].
```

[← Previous: Appendix Index](../appendix.md) | [Table of contents](../main.md#chapters) | [Next: Bellman Expectation Derivation](b-bellman-expectation-derivation.md) →
