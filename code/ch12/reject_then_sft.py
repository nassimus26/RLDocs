def f(y):
    r_acc = 1.0 if y[-1] == ANS else 0.0
    r_format = 1.0 if y[0] == VOCAB.index("it") else 0.0
    return 1.0 if r_acc == 1.0 and r_format == 1.0 else 0.0

def reject_then_sft(pi, pi_base, d_gen=(), n_y=4, alpha=0.05, epochs=20):
    # 1. D_keep <- ?
    d_keep = []
    # 2. For many x ? D: sample y ? pi(*|x); keep if f=1
    for x in PROMPTS:
        for _ in range(n_y):
            y = sample_response(pi, x)  # y ? pi(*|x); pi is the GRPO actor
            if f(y) == 1:
                d_keep.append((x, y))
    # 3. D_SFT <- D_keep ? D_gen
    d_sft = d_keep + list(d_gen)
    # 4-6. pi_theta <- pi_base; teacher forcing on D_SFT
    return sft(pi_base, d_sft, alpha=alpha, epochs=epochs)
