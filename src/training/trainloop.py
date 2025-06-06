import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from training.trajectorycollector import generate_trajectory
from training.ppotrainer import ppo_update
import random


# Main training loop for PPO applied to Wordle
# made in part with generative AI
def training_loop(
    env,                    # the environment
    observation_encoder,      # Encodes game state into model-friendly tensors
    shared_encoder,           # Shared encoder for feature extraction
    policy_head,              # Predicts best word to guess
    value_head,               # Estimates value function V(s)
    word_encoder,             # Encodes words into embeddings
    optimizer_policy,         # Optimizer for the policy network
    optimizer_value,          # Optimizer for the value network
    word_list,                    # List of all valid guess words
    num_epochs=1000,          # Number of PPO update rounds
    episodes_per_epoch=16,       # Number of episodes to simulate before each PPO update
    ppo_epochs=4,             # Number of passes over the same data in PPO
    minibatch_size=32,        # Batch size for PPO updates
    gamma=1.0,                # Discount factor (can be 1.0 for episodic Wordle)
    clip_epsilon=0.2,         # PPO clipping parameter
    device=torch.device("cpu") # Device to run training on
):
    # Function to regenerate word embeddings each epoch (they're trainable)
    def generate_word_embeddings():
        return torch.stack([word_encoder(word) for word in word_list], dim=0).to(device)
    
    word_embeddings = generate_word_embeddings()

    for epoch in trange(num_epochs, desc="Training"):

        # Storage for all transitions from this batch of episodes
        all_obs, all_actions, all_old_log_probs, all_returns, all_advantages = [], [], [], [], []

        # Collect episodes_per_epoch number of episodes (trajectories)
        for _ in range(episodes_per_epoch):
            traj = generate_trajectory(
                env,
                word_list,
                observation_encoder,
                shared_encoder,
                policy_head,
                value_head,
                word_embeddings,
                device=device,
                gamma=gamma
            )
            all_obs.extend(traj["observations"])
            all_actions.extend(traj["actions"])
            all_old_log_probs.extend(traj["log_probs"])
            all_returns.extend(traj["returns"])
            all_advantages.extend(traj["advantages"])

        # Normalize advantages (in-place)
        adv_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        mean, std = adv_tensor.mean(), adv_tensor.std()
        all_advantages = [(a - mean) / (std + 1e-8) for a in all_advantages]

        # Zip into dataset (leave all as raw Python objects or lightweight tensors)
        dataset = list(zip(all_obs, all_actions, all_old_log_probs, all_returns, all_advantages))
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        # PPO epochs: sample random batches from dataset each pass
        for _ in range(ppo_epochs):
            # Shuffle indices for each epoch
            random.shuffle(indices)
        
            # Process minibatches
            for start_idx in range(0, dataset_size, minibatch_size):
                batch_indices = indices[start_idx:start_idx + minibatch_size]
            
                # Extract batch data
                obs_batch = [dataset[i][0] for i in batch_indices]
                act_batch = [dataset[i][1] for i in batch_indices]
                logp_batch = [dataset[i][2] for i in batch_indices]
                ret_batch = [dataset[i][3] for i in batch_indices]
                adv_batch = [dataset[i][4] for i in batch_indices]
                
                # Recompute embeddings so gradients flow through word_encoder
                word_embeddings = generate_word_embeddings()
            
                #torch.autograd.set_detect_anomaly(True)
                # Perform PPO update
                print("update")
                ppo_update(
                    observation_encoder,
                    shared_encoder,
                    policy_head,
                    value_head,
                    optimizer_policy,
                    optimizer_value,
                    obs_batch,
                    act_batch,
                    adv_batch,
                    ret_batch,
                    logp_batch,
                    word_embeddings,
                    clip_epsilon=clip_epsilon,
                    device=device
                )
