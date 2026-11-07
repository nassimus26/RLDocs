# Reinforcement Learning:
Mathematics and algorithms from MDPs to RLHF, InstructGPT,
and DeepSeek-R1, with Python

Welcome to the open-source repository for **Reinforcement Learning: Mathematics and algorithms from MDPs to RLHF, InstructGPT, and DeepSeek-R1, with Python**. This comprehensive textbook serves as an educational compilation and rigorous distillation of the mathematically established methods, architectures, and algorithms that define modern Reinforcement Learning (RL).

Rather than proposing speculative novel inventions, this book acts as a clean, self-contained, and structurally cohesive reference manual designed to bridge the gap between abstract mathematical proofs and concrete programmatic implementations.

---

## 🎯 Target Audience & Purpose

This book is deliberately designed for:
* **Machine Learning Engineers & Data Scientists:** Looking to master the mathematical mechanics behind the algorithms they deploy (e.g., PPO, DeepSeek-R1, and Actor-Critic variants).
* **Academic Students & Researchers:** Seeking a rigorous, unified mathematical framework that systematically transitions from foundational Markov Decision Processes (MDPs) to contemporary LLM reinforcement learning alignment.
* **Self-Directed Learners:** Who want a self-contained guide covering both classical tabular theory and deep, scalable function approximation without skipping steps in the derivations.

---

## ✨ Key Highlights & Features

* **Complete Mathematical Derivations:** No "hand-waving." Every equation—from Bellman Optimality and the Robbins-Monro convergence criteria to policy gradient updates—is derived step-by-step.
* **Unified Architectural Pacing:** Progresses logically from basic environments to the complex reward-shaping mechanics underpinning state-of-the-art modern reasoning systems (such as DeepSeek-R1, GRPO, and RLHF frameworks).
* **Worked Code Examples:** Each algorithm is paired with a small, readable Python/PyTorch implementation on toy environments (e.g., GridWorld, CartPole), so the derivations and the code can be read side by side.

---

## 📂 Available Formats

This book is available in the following formats in this repository:

* **📄 [PDF](tex/main.pdf):** The compiled publication-ready PDF for reading and printing.
* **📝 [Markdown](md/main.md):** The main book source in Markdown format for browser viewing and lightweight editing.
* **📘 [LaTeX source](tex/main.tex):** The full source used to generate the PDF, including the chapter structure and publication layout.

---

## 🛠️ Methodology & Curation Notice

This text openly declares its development methodology:

* **Linguistic Refinement:** Generative artificial intelligence tools were utilized to refine the prose explanations of these standard methods and optimize their linguistic flow.
* **Authorial Governance:** The final text, overarching structure, and conceptual curation were executed by the author. All mathematical derivations, algorithmic loops, and Python source code listings have been manually cross-verified against primary literature by the author, who assumes sole responsibility for the technical correctness, accuracy, and integrity of the final work.

---

## 📚 Repository Structure

```text
├── README.md                  # Project overview and usage notes
├── .gitignore                 # Ignore rules for local/generated files
├── code/                      # RL implementations and chapter examples
├── figures/                   # Figure assets used by the book and docs
├── md/                        # Markdown source for chapters and narrative content
├── scripts/                   # Export and build scripts
├── tex/                       # LaTeX source for publication workflows
```

---

## 🤝 Citation & Open Access

This textbook is an open-access resource licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to share, adapt, and build upon this material for any purpose, provided that you give appropriate credit to the author and cite the permanent archive DOI.

To cite this book in your academic work, please use the following BibTeX entry:

```bibtex
@book{gautam2026rl,
  author    = {Gautam, Dipesh},
  title     = {Reinforcement Learning: Mathematics and algorithms from MDPs to RLHF, InstructGPT, and DeepSeek-R1, with Python},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23996511},
  url       = {https://doi.org/10.5281/zenodo.23996511},
  note      = {Open-source textbook available at \url{https://github.com/dipeshgautam2012/rl-book-project}}
}
```