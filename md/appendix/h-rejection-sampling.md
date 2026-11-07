[← Previous: PPO Derivations](g-ppo-derivations.md) | [Table of contents](../main.md#chapters) | [Next: References](../references.md) →

# Appendix H: Rejection Sampling

**Rejection sampling** produces samples from a **target** $p$ when you can only draw from a **proposal** $q$ that is easy to sample. It does not reweight those draws. It **accepts or drops** each one, using a keep-chance $a$ and a die, until the accepted batch has law $p$.

## H.1 Example

You are filling a party bowl. Beans must meet hidden requirements (taste, flavor, hardness, ...), but you cannot test those when picking --- you only see **color**. From color alone you guess the chance a bean meets those requirements: red $80\%$, green $60\%$, blue $40\%$, yellow $20\%$. Those guesses are the keep-chances $a$. You also care about **color composition in the bowl**: $40\%$ red, $30\%$ green, $20\%$ blue, $10\%$ yellow. That desired composition is the target $p$.

The jar has infinitely many beans, equally mixed. Drawing one at random, each color has probability $\tfrac14$ on every draw (a finite jar of $100$ with $25$ of each would change those odds after each draw if you do not replace). That unchanging draw distribution is the proposal $q$. It is **not** the bowl composition you want: 

```math
q(\mathrm{red})=q(\mathrm{green})=q(\mathrm{blue})=q(\mathrm{yellow})=\tfrac14,
```

```math
p(\mathrm{red})=40\%,\quad p(\mathrm{green})=30\%,\quad p(\mathrm{blue})=20\%,\quad p(\mathrm{yellow})=10\%,
```

```math
a(\mathrm{red})=80\%,\quad a(\mathrm{green})=60\%,\quad a(\mathrm{blue})=40\%,\quad a(\mathrm{yellow})=20\%.
```

 The die is compared to $a$, not to $p$. $q$ is the jar-draw probability.

- Pick one bean from the jar --- say **red**.
- Look up $a(\mathrm{red})=80\%$.
- Roll a $100$-sided die --- get $37$.
- $37\%\le 80\%$ → treat it as ``passes,'' put it in the **bowl**.

Draw again and get **yellow**: guess $20\%$, roll $40$ → $40\%\le 20\%$? **no** → reject, pick again. Repeat. The bowl approaches the composition you wanted ($p$).

## H.2 Keep-chance from counts

Write $c$ for a color. Draw $N$ beans from the jar. About $N\,q(c)$ of them are color $c$. Of those you keep a fraction $a(c)$, so the count of kept beans of color $c$ is $N\,q(c)\,a(c)$. The bowl only has kept beans. Total kept is the same count, summed over all colors ($c'=$ red, green, blue, yellow): 

```math
\sum_{c'} N\,q(c')\,a(c')=N\sum_{c'} q(c')\,a(c').
```

 The fraction of color $c$ in the bowl is (kept $c$) / (total kept): 

```math
\frac{N\,q(c)\,a(c)}{N\sum_{c'} q(c')\,a(c')}=\frac{q(c)\,a(c)}{\sum_{c'} q(c')\,a(c')}.
```

 $N$ cancels. The denominator is the overall keep rate $P(\mathrm{accept})=\sum_{c'} q(c')\,a(c')$.

**This example, $N=100$.** Jar equal mix: about $25$ of each color. Keep $25\times 80\%=20$ red, $15$ green, $10$ blue, $5$ yellow. Bowl size $50$. Fraction red $=20/50=40\%$, which is $p(\mathrm{red})$. Same pattern for the others.

You want that bowl fraction to equal the target: 

```math
\frac{q(c)\,a(c)}{P(\mathrm{accept})}=p(c)\qquad\Rightarrow\qquad a(c)=\frac{p(c)\,P(\mathrm{accept})}{q(c)}.
```

 $P(\mathrm{accept})$ is the same for every color. Call it $1/M$ (so $M=1/P(\mathrm{accept})\ge 1$). Then 

```math
a(c)=\frac{p(c)}{M\,q(c)}.
```

 Need $0\le a(c)\le 1$ for every $c$, so $M$ must satisfy $p(c)\le M q(c)$ everywhere; the smallest such choice is $M=\max_c p(c)/q(c)$. Any larger bound also works.

**This example.** $q(c)=\tfrac14$ for every color and $p=(0.40,0.30,0.20,0.10)$, so $\max p/q=0.40/(1/4)=1.6$. Pick $M=2$ (any $M\ge 1.6$ works; $2$ keeps arithmetic simple). Then $M q=\tfrac12$ and 

```math
a(\mathrm{red})=\frac{0.40}{0.50}=0.80,\;a(\mathrm{green})=0.60,\;a(\mathrm{blue})=0.40,\;a(\mathrm{yellow})=0.20.
```

 Overall keep rate $P(\mathrm{accept})=1/M=\tfrac12$, which matches $\sum q\,a=\tfrac14(0.8+0.6+0.4+0.2)=\tfrac12$. Using the tight $M=1.6$ instead would give $a=(100\%,\,75\%,\,50\%,\,25\%)$.

**Loop.** Given a way to evaluate $p(x)$ and $q(x)$, sample $X\sim q$, and sample $U\sim\mathrm{Unif}(0,1)$ (the die --- not a draw from $a$ itself):

- Compute $M=\max_x p(x)/q(x)$ (or any larger bound), then $a(x)=p(x)/(M q(x))$ as in (H.2).
- Sample $X\sim q$.
- Sample $U\sim\mathrm{Unif}(0,1)$.
- Accept if $U\le a(X)$; else retry.

The accepted $X$ has law $p$.

## H.3 Hard filter

Same rejection loop, but the keep-chance is only $0$ or $1$: $a(\cdot)\in\lbrace0,1\rbrace$. No die and no $U$. Keep for sure or drop for sure.

Let $\pi(y\mid x)$ be easy to sample. Filter $f(x,y)\in\lbrace0,1\rbrace$. That filter **is** the keep-chance: $a(y)=f(x,y)$. Copy (H.1) with $q\to\pi(\cdot\mid x)$, $a\to f$, and color $c\to$ reply $y$ (prompt $x$ fixed): 

```math
p(y\mid x)=\frac{\pi(y\mid x)\,f(x,y)}{\sum_{y'} \pi(y'\mid x)\,f(x,y')}.
```

 The denominator is the overall keep rate (pass rate). Write $Z(x)$ for it. Because $f$ is $0$ or $1$, that keep rate is just $P(f=1\mid x)$: 

```math
Z(x)=\sum_{y'} \pi(y'\mid x)\,f(x,y')=P(f=1\mid x).
```

 So the same bowl formula is 

```math
p(y\mid x)=\frac{\pi(y\mid x)\,f(x,y)}{Z(x)}.
```

 If $f=0$, the numerator is $0$, so that $y$ is $0\%$ of the bowl. If $f=1$, $p(y\mid x)=\pi(y\mid x)/Z(x)$ --- the same relative mix as $\pi$, among passers only. This is rejection sampling with $a\in\lbrace0,1\rbrace$ and $M=1/Z(x)$. Accepted $y\sim\pi(\cdot\mid x,\,f=1)$.

Toy: four replies, $\pi=(0.40,0.10,0.30,0.20)$, $f=(1,1,0,0)$. Denominator $Z=0.40+0.10+0+0=0.50$. Then $p=(0.80,0.20,0,0)$. Same shape as $N=100$ beans: keep $40$ and $10$, drop $30$ and $20$, bowl size $50$, fractions $40/50$ and $10/50$.

The loop **never uses** $Z$. The die is compared to $a$, not to the overall keep rate; $Z$ is that overall keep rate, so it is not the cutoff. Here $a=f\in\lbrace0,1\rbrace$, so you skip the die ($f=1$ always keep, $f=0$ always drop). The steps only sample from $\pi$ and check $f$.

[← Previous: PPO Derivations](g-ppo-derivations.md) | [Table of contents](../main.md#chapters) | [Next: References](../references.md) →
