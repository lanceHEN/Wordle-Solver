import torch
from torch.distributions import Categorical

# made with help of generative AI
def generate_trajectory(env, word_list, observation_encoder, shared_encoder, policy_head, value_head, word_embeddings, device="cpu", gamma=1):
    """
    Simulates one episode of Wordle using the current policy with embedding-based scoring.

    At each step:
    - Encodes the observation.
    - Computes a query embedding and scores all words via dot product, masking invalid ones.
    - Samples an action after taking softmax over the scores, logs probability, and records reward and value.
    
    Returns:
        A dictionary with everything needed for PPO training - observations, actions, log probs, returns, advantages, and indices of valid words.
    """
    obs = env.reset()
    done = False

    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    valid_indices_all = []

    with torch.no_grad():
        while not done:
            # Encode observation
            grid_tensor, meta_tensor = observation_encoder(obs)
            grid_tensor = grid_tensor.to(device)
            meta_tensor = meta_tensor.to(device)

            # get latent state
            h_policy = shared_encoder(grid_tensor.unsqueeze(0), meta_tensor.unsqueeze(0))
            h_value = shared_encoder(grid_tensor.unsqueeze(0), meta_tensor.unsqueeze(0))

            # Get valid action logits (from dot product )
        
            valid_indices = obs["valid_indices"]
            scores = policy_head(h_policy, [valid_indices], word_embeddings).clone().detach().requires_grad_(True) # logits for all guessses - [1, vocab_size]
            dist = Categorical(logits=scores)
            # Choose an action
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx) # get log probability - taking the log provides numerical stability

            # Get actual word from index
            action_word = word_list[action_idx.item()]

            # Step the environment
            next_obs, reward, done = env.step(action_word)

            value = value_head(h_value).squeeze() # get predicted value for state

            # Store trajectory
            observations.append(obs)
            actions.append(action_idx.item())
            log_probs.append(log_prob.squeeze(0))
            rewards.append(torch.tensor(reward, dtype=torch.float))
            values.append(value)
            valid_indices_all.append(valid_indices)

            obs = next_obs
            #print(obs["valid_indices"])

        # final value - always 0 because of terminal state
        last_value = 0

        # Advantage and return calculations
        values.append(torch.tensor(last_value))
        advantages, returns = compute_advantages(rewards, values, gamma=gamma)


        return {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "valid_indices": valid_indices_all,
            "values": values
        }

# made with help of generative AI
def compute_advantages(rewards, values, gamma=1):
    """
    Computes simple advantages and returns.
    Advantages are simply the difference of returns and predicted values.
    
    Args:
        rewards: list of individual rewards [r_0, r_1, ..., r_T-1]
        values: list of value estimates [v_0, v_1, ..., v_T] (note: T+1 entries)
        gamma: discount factor
    
    Returns:
        advantages: A_t = G_t - V(s_t)
    """
    returns = []
    G = values[-1].item()  # start with bootstrapped value 0 
    for r in reversed(rewards): # work backwards from end, accumulating actual rewards
        G = r.item() + gamma * G
        returns.insert(0, G)

    # Convert to tensors
    returns = torch.tensor(returns, dtype=torch.float)
    values = torch.stack(values[:-1])
    advantages = returns - values

    return advantages, returns