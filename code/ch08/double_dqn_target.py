    with torch.no_grad():
    # y_j <- r_j + gamma(1-d_j) Q(s'_j, argmax_a' Q(s'_j,a';theta); theta^-)
    best = theta(next_states).argmax(dim=1)
    y = rewards + gamma * (1.0 - dones) * theta_(next_states)[range(batch_size), best]
