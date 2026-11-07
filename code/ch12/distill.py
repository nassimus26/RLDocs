def distill(pi_base, d_sft, alpha=0.05, epochs=20):
    # 1-3. pi_theta <- pi_base (student); teacher forcing on D_SFT already kept
    return sft(pi_base, d_sft, alpha=alpha, epochs=epochs)
   
