def sft(pi_base, demos, alpha=0.05, epochs=20):
    # 1. theta <- theta_base
    h = copy.deepcopy(pi_base)
    opt = torch.optim.SGD(h.parameters(), lr=alpha)
    for _ in range(epochs):
        for x, y in demos:  # gold (x, y) ? D_SFT
            loss = -log_pi(h, x, y)  # L_SFT on one pair (12.2)
            opt.zero_grad()
            loss.backward()
            opt.step()  # (12.4)
    return h
