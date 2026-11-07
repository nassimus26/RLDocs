def rlhf(pi_ref, r, beta=0.1, K=4, alpha_theta=0.05, max_iters=1000, E=4, eps=0.2):
    # 3. Initialize theta from pi_ref
    h = copy.deepcopy(pi_ref)
    opt = torch.optim.SGD(h.parameters(), lr=alpha_theta)
    for _ in range(max_iters):  # 4. Loop for each iteration
        x = random.choice(PROMPTS)
        ys = [sample_response(h, x) for _ in range(K)]  # y_k ? pi_theta_old(*|x)
        with torch.no_grad():
            logp_old = [log_pi(h, x, y) for y in ys]  # log pi_theta_old(y_k|x)
            # R_k <- r_phi(x,y_k) ? ? (log pi_theta(y_k|x) ? log pi_ref(y_k|x))  (11.4)
            R = [r_phi(r, x, y).item() - beta * (lp.item() - log_pi(pi_ref, x, y).item())
                 for y, lp in zip(ys, logp_old)]
        # A_k <- R_k ? b_k; keep A_k fixed for this outer round
        A = [R[k] - (sum(R) - R[k]) / (K - 1) for k in range(K)]

        for _ in range(E):
            logps = [log_pi(h, x, y) for y in ys]
            rhos = [torch.exp(lp - lo) for lp, lo in zip(logps, logp_old)]
            # L^PPO <- ?(1/K) sum_k min(?_k A_k, clip(?_k, 1?eps, 1+eps) A_k)  (11.9)
            loss = -sum(
                torch.min(rho * A[k], torch.clamp(rho, 1 - eps, 1 + eps) * A[k])
                for k, rho in enumerate(rhos)
            ) / K
            opt.zero_grad()
            loss.backward()
            opt.step()
    return h  # 5. Return pi_theta
