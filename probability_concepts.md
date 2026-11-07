> **RL book:** edit [`book.md`](book.md) in Cursor.
> For Cursor-like math + algorithm boxes on the web: `python3 scripts/build_pages.py` → open [`docs/index.html`](docs/index.html)
> (uses KaTeX + `github-markdown-light.css`). Raw github.com Markdown cannot look the same — it uses MathJax and ignores custom CSS.
> Optional: make the repo public and enable Pages from `/docs` for https://dipeshgautam2012.github.io/reinforcement_learning/

# Law of Total Probability & Bayes' Theorem

<img src="data:image/svg+xml;utf8,<svg width='900' height='420' viewBox='0 0 900 420' xmlns='http://www.w3.org/2000/svg'><rect x='50' y='50' width='800' height='230' fill='white' stroke='black' stroke-width='3'/><text x='50' y='38' font-size='26' font-weight='bold'>Ω</text><line x1='250' y1='50' x2='250' y2='280' stroke='black' stroke-width='2' stroke-dasharray='8,8'/><line x1='450' y1='50' x2='450' y2='280' stroke='black' stroke-width='2' stroke-dasharray='8,8'/><line x1='650' y1='50' x2='650' y2='280' stroke='black' stroke-width='2' stroke-dasharray='8,8'/><text x='135' y='90' font-size='26'>B₁</text><text x='335' y='90' font-size='26'>B₂</text><text x='535' y='90' font-size='26'>B₃</text><text x='735' y='90' font-size='26'>B₄</text><path d='M 80 145 C 100 115, 150 115, 200 130 C 270 145, 320 120, 400 135 C 480 150, 530 120, 600 135 C 680 150, 750 115, 810 145 C 860 175, 830 220, 780 225 C 700 235, 650 210, 580 225 C 500 240, 450 215, 380 225 C 300 235, 230 215, 160 225 C 100 230, 60 190, 80 145 Z' fill='%23ffcccc' fill-opacity='0.7' stroke='red' stroke-width='3'/><text x='20' y='185' font-size='30' fill='red'>A</text><text x='110' y='185' font-size='25' fill='darkred'>A ∩ B₁</text><text x='310' y='185' font-size='25' fill='darkred'>A ∩ B₂</text><text x='510' y='185' font-size='25' fill='darkred'>A ∩ B₃</text><text x='710' y='185' font-size='25' fill='darkred'>A ∩ B₄</text><text x='115' y='330' font-size='23' fill='blue'>P(B₁)</text><text x='315' y='330' font-size='23' fill='blue'>P(B₂)</text><text x='515' y='330' font-size='23' fill='blue'>P(B₃)</text><text x='715' y='330' font-size='23' fill='blue'>P(B₄)</text></svg>" alt="Law of Total Probability Diagram" />

## 1. Law of Total Probability

Expressing event $A$ as a union of disjoint intersections:

```math
A=(A\cap B_1)\cup(A\cap B_2)\cup(A\cap B_3)\cup\cdots\cup(A\cap B_n)
```

Summing the probabilities of these disjoint events:

```math
P(A)=P(A\cap B_1)+P(A\cap B_2)+P(A\cap B_3)+\cdots+P(A\cap B_n)
```
Hence,
```math
\boxed{
P(A)=\sum_{i=1}^{n}P(A\cap B_i)
}
```

Using the multiplication rule,

```math
P(A\cap B_i)=P(A\mid B_i)P(B_i)
```

we obtain:

```math
\boxed{
P(A)=\sum_{i=1}^{n}P(A\mid B_i)P(B_i)
}
```

---

## 2. Bayes' Theorem

By the definition of conditional probability:

```math
P(B_k\mid A)=\frac{P(A\cap B_k)}{P(A)}
```

Applying the multiplication rule to the numerator:

```math
P(A\cap B_k)=P(A\mid B_k)P(B_k)
```

Substituting the Law of Total Probability into the denominator:

```math
\boxed{
P(B_k\mid A)=
\frac{P(A\mid B_k)P(B_k)}
{\displaystyle\sum_{i=1}^{n}P(A\mid B_i)P(B_i)}
}
```

# Conditional Expectation & The Law of Total Expectation

## 1. Conditional Expectation $E[X\mid Y]$

### Mathematical Definition & Derivation

Let $X$ and $Y$ be discrete random variables. For a fixed value $Y=y$ where $P(Y=y)>0$, the conditional expectation of $X$ given $Y=y$ is the expected value of $X$ under the conditional probability distribution $P(X=x\mid Y=y)$:

```math
E[X\mid Y=y]=\sum_x xP(X=x\mid Y=y)
```

Using the definition of conditional probability:

```math
P(X=x\mid Y=y) =
\frac{P(X=x,Y=y)}{P(Y=y)}
```

this can be written explicitly as:

```math
E[X\mid Y=y] =
\sum_x x
\frac{P(X=x,Y=y)}{P(Y=y)}
```

Because $E[X\mid Y=y]$ produces a numerical value for each possible value $y$, $E[X\mid Y]$ is itself a **random variable** that is a function of $Y$, written as $g(Y)$.

---

### Example: Computing $E[X\mid Y]$

Consider a set of **10 balls, big and small, with three colors: red (R), green (G), and blue (B)**, categorized by two discrete random variables:

* **Color ($X$):** $X\in\{R,G,B\}$
* **Size ($Y$):** $Y\in\{\text{Large},\text{Small}\}$

### Joint Distribution Table

| Color ($X$) | Large ($Y=\text{Large}$) | Small ($Y=\text{Small}$) | Total |
|:---|:---:|:---:|:---:|
| **$X=R$** | 1 | 2 | **3** |
| **$X=G$** | 3 | 1 | **4** |
| **$X=B$** | 2 | 1 | **3** |
| **Total** | **6** | **4** | **10** |

### Step-by-Step Calculation for Each Outcome of $Y$

To evaluate the weighted sum, we assign numerical values to the colors:

**$R=1$, $G=2$, and $B=3$.**

#### 1. For $Y=\text{Large}$ (6 total balls)

* $P(X=R\mid Y=\text{Large})=\frac{1}{6}$
* $P(X=G\mid Y=\text{Large})=\frac{3}{6}$
* $P(X=B\mid Y=\text{Large})=\frac{2}{6}$

```math
E[X\mid Y=\text{Large}] =
\left(R\cdot\frac{1}{6}\right) +
\left(G\cdot\frac{3}{6}\right) +
\left(B\cdot\frac{2}{6}\right)
=
\left(1\cdot\frac{1}{6}\right) +
\left(2\cdot\frac{3}{6}\right) +
\left(3\cdot\frac{2}{6}\right) =
\frac{1+6+6}{6} =
\frac{13}{6}
\approx2.167
```

#### 2. For $Y=\text{Small}$ (4 total balls)

* $P(X=R\mid Y=\text{Small})=\frac{2}{4}$
* $P(X=G\mid Y=\text{Small})=\frac{1}{4}$
* $P(X=B\mid Y=\text{Small})=\frac{1}{4}$

```math
E[X\mid Y=\text{Small}] =
\left(R\cdot\frac{2}{4}\right) +
\left(G\cdot\frac{1}{4}\right) +
\left(B\cdot\frac{1}{4}\right)
=
\left(1\cdot\frac{2}{4}\right) +
\left(2\cdot\frac{1}{4}\right) +
\left(3\cdot\frac{1}{4}\right) =
\frac{2+2+3}{4} =
\frac{7}{4} =
1.75
```

Thus, the random variable $E[X\mid Y]$ takes the value $\frac{13}{6}$ when $Y=\text{Large}$ and $\frac{7}{4}$ when $Y=\text{Small}$.

---

## 2. **Law of Total Expectation**

The expected value of the conditional expectation of $X$ given $Y$ is the same as the expected value of $X$:

```math
E[X]=E[E[X\mid Y]]
```

**Proof:**

```math
E[X]=\sum_x xP(X=x)
```

From the **Law of Total Probability**:

```math
P(X=x) =
\sum_yP(X=x\mid Y=y)P(Y=y)
```

Therefore:

```math
E[X] =
\sum_x x
\left[
\sum_yP(X=x\mid Y=y)P(Y=y)
\right]
=
\sum_y
\sum_x
xP(X=x\mid Y=y)P(Y=y)
=
\sum_y
\left[
\sum_xxP(X=x\mid Y=y)
\right]
P(Y=y)
```

By the definition of conditional expectation:

```math
\sum_xxP(X=x\mid Y=y) =
E[X\mid Y=y]
```

Therefore:

```math
\boxed{
E[X] =
\sum_yE[X\mid Y=y]P(Y=y)
}
```

Notice that for a fixed value $Y=y$, $E[X\mid Y=y]$ produces a specific constant value—for example, $\frac{13}{6}$ when $Y=\text{Large}$ and $\frac{7}{4}$ when $Y=\text{Small}$ in the example above.

Because this value changes depending on $y$, $E[X\mid Y]$ is a function of $Y$, which we can write as:

```math
g(Y)=E[X\mid Y]
```

Since $Y$ is random, $g(Y)$ is itself a **random variable**.

Hence:

```math
E[X] =
\sum_y g(Y)P(Y=y)
=
E[g(Y)]
```

Therefore:

```math
\boxed{
E[X]=E[E[X\mid Y]]
}
```

---

### Verification Using the Ball Example

Using the calculated values and probabilities:

* $E[X\mid Y=\text{Large}]=\frac{13}{6}$ with $P(Y=\text{Large})=\frac{6}{10}$
* $E[X\mid Y=\text{Small}]=\frac{7}{4}$ with $P(Y=\text{Small})=\frac{4}{10}$

**1. Left-hand side $E[E[X\mid Y]]$:**

```math
E[E[X\mid Y]] =
\left(\frac{13}{6}\cdot\frac{6}{10}\right) +
\left(\frac{7}{4}\cdot\frac{4}{10}\right)
=
\frac{13}{10} +
\frac{7}{10} =
\frac{20}{10} =
2
```

**2. Right-hand side $E[X]$:**

* $P(X=R)=\frac{3}{10}$
* $P(X=G)=\frac{4}{10}$
* $P(X=B)=\frac{3}{10}$

```math
E[X] =
\left(1\cdot\frac{3}{10}\right) +
\left(2\cdot\frac{4}{10}\right) +
\left(3\cdot\frac{3}{10}\right)
=
\frac{3+8+9}{10} =
\frac{20}{10} =
2
```

Both sides equal $2$, confirming:

```math
\boxed{
E[E[X\mid Y]]=E[X]
}
```

# Intuition Behind the Law of Total Expectation: $E[E[X\mid Y]]=E[X]$

The Law of Total Expectation states:

```math
\boxed{
\text{Overall average} =
\text{weighted average of group averages}
}
```

### Concrete Group Breakdown

Suppose the population is divided into groups $A$, $B$, $C$, $\dots$

- **Group A:** $N_a$ values with average $\bar a$
- **Group B:** $N_b$ values with average $\bar b$
- **Group C:** $N_c$ values with average $\bar c$

The total population is:

```math
N_{\text{total}}=N_a+N_b+N_c+\dots
```

### Step 1: Overall Average

Calculate the average using all values directly:

```math
E[X] =
\frac{
\text{sum of all values}
}{
N_{\text{total}}
}
```

### Step 2: Group Averages

Calculate the average within each group:

```math
\bar a=\frac{a_1+a_2+\dots+a_{N_a}}{N_a}
```

```math
\bar b=\frac{b_1+b_2+\dots+b_{N_b}}{N_b}
```

```math
\bar c=\frac{c_1+c_2+\dots+c_{N_c}}{N_c}
```

### Step 3: Weighted Average of Group Averages

Weight each group average by its proportion of the total population:

```math
E[E[X\mid\text{Group}]] =
\bar a\left(\frac{N_a}{N_{\text{total}}}\right) +
\bar b\left(\frac{N_b}{N_{\text{total}}}\right) +
\bar c\left(\frac{N_c}{N_{\text{total}}}\right)
```

Substituting the group averages, the group sizes cancel:

```math
E[E[X\mid\text{Group}]] =
\frac{
(a_1+\dots+a_{N_a}) +
(b_1+\dots+b_{N_b}) +
(c_1+\dots+c_{N_c})
}{
N_{\text{total}}
}
```

This is exactly the overall average:

```math
\boxed{
E[E[X\mid\text{Group}]]=E[X]
}
```

> **Key idea:**  
> First, find the average **within each group**.  
> Then, take the **weighted average of those group averages**.  
> The result is the **overall average**.

```math
\boxed{
E[X] =
\sum_y E[X\mid Y=y]P(Y=y)
}
```

```math
\boxed{
E[E[X\mid Y]]=E[X]
}
```