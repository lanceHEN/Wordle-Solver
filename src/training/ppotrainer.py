import torch
import torch.nn.functional as F

# made in part with generative AI
def ppo_update(
    observation_encoder,
    shared_encoder,
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    observations,
    actions,
    advantages,
    returns,
    old_log_probs,
    word_embeddings,
    clip_epsilon=0.2,
    device="cpu"
):
    """
    Performs a single PPO update step over the given trajectory, with the given observations, rewards, etc to batch.
    For positive advantage (scoring higher than expected), we want to encourage a higher probability of that action.
    For negative advantage (scoring lower than expected), we want to encourage a lower probability of that action.
    """
    # Clear gradients explicitly
    optimizer_policy.zero_grad(set_to_none=True)
    optimizer_value.zero_grad(set_to_none=True)
    
    # Encode all observations
    grids, metas, valid_indices_batch = [], [], []
    for obs in observations:
        grid, meta = observation_encoder(obs)
        grids.append(grid)
        metas.append(meta)
        valid_indices_batch.append(obs["valid_indices"])

    # Turn lists into tensors
    grids = torch.stack(grids).to(device)
    metas = torch.stack(metas).to(device)
    actions = torch.tensor(actions, device=device)
    advantages = torch.tensor(advantages, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    old_log_probs = torch.stack(old_log_probs).to(device)

    # Forward pass
    h_policy = shared_encoder(grids, metas)  # [B, hidden_dim]
    h_value = shared_encoder(grids, metas)  # [B, hidden_dim]
    values = value_net(h_value)  # [B]

    logits = policy_net(h_policy, valid_indices_batch, word_embeddings).clone().detach().requires_grad_(True)  # [B, vocab_size]
    print("[ppo_update] after policy_head: query-related logits shape:", logits.shape)

    log_probs = F.log_softmax(logits, dim=-1)
    taken_log_probs = log_probs[torch.arange(len(actions)), actions].clone()

    # PPO loss
    ratios = torch.exp(taken_log_probs - old_log_probs.detach()) # new vs old
    #print(torch.max(logits, dim=1))
    #print([len(x) for x in valid_indices_batch])
    print(taken_log_probs)
    #print(old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    value_loss = F.mse_loss(values, returns)
    
    total_loss = policy_loss + value_loss

    # Backward + optimize
    #torch.autograd.set_detect_anomaly(True) # for debugging
    print("[ppo_update] about to call backward()")
    total_loss.backward()
    
    optimizer_policy.step()
    optimizer_value.step()
