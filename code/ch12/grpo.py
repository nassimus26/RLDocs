def r_rule(y):
    r_acc = 1.0 if y[-1] == ANS else 0.0
    r_format = 1.0 if y[0] == VOCAB.index("it") else 0.0
    return r_acc + r_format  # (12.5)

def grpo(pi_start, beta=0.1, G=4, alpha=0.05, max_iters=1000, E=4, eps=0.2, delta=1e-8):
    # 1. pi_ref <- pi_theta (freeze)
    h = copy.deepcopy(pi_start)
    pi_ref = copy.deepcopy(h)
    opt = torch.optim.SGD(h.parameters(), lr=alpha)
    for _ in range(max_iters):
        x = random.choice(PROMPTS)
        ys = [sample_response(h, x) for _ in range(G)]  # y^(i) ? pi_theta_old
        with torch.no_grad():
            logp_old = [log_pi(h, x, y) for y in ys]
            logp_ref = [log_pi(pi_ref, x, y) for y in ys]
            r = [r_rule(y) for y in ys]  # (12.5); no r_phi, no KL
            mu = sum(r) / G
            sd = (sum((ri - mu) ** 2 for ri in r) / G) ** 0.5
            A = [(ri - mu) / (sd + delta) for ri in r]  # (12.6)
        for _ in range(E):
            logps = [log_pi(h, x, y) for y in ys]
            rhos = [torch.exp(lp - lo) for lp, lo in zip(logps, logp_old)]
            clip = -sum(
                torch.min(rho * A[i], torch.clamp(rho, 1 - eps, 1 + eps) * A[i])
                for i, rho in enumerate(rhos)
            ) / G
            kl = sum(lp - lr for lp, lr in zip(logps, logp_ref)) / G
            loss = clip + beta * kl  # (12.7)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return h
