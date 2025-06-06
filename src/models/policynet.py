import torch
import torch.nn as nn

# a policy network that takes in the latent vector for the current game from SharedEncoder and produces logits
# over every single word in the vocabulary of possible guesses
# logits belonging to invalid words (words that cannot possibly be a guess, given the feedback) are set to -inf so they are not considered
# one can apply softmax over logits to get a probability distribution to either sample from or take the argmax of, depending on whether in training or eval
# this is a simple MLP with one linear layer
class PolicyHead(nn.Module):
    
    # Initializes a PolicyHead with the given input latent dimension (should be same as SharedEncoder output), vocab size, and device
    def __init__(self, hidden_dim, vocab_size, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.policy = nn.Linear(hidden_dim, vocab_size) # linear layer
    
    # given a latent vector for the current game from SharedEncoder, as well as the indices of remaining valid guesses,
    # produces logits over every single word in the vocabulary of possible guesses
    # logits belonging to invalid words (words that cannot possibly be a guess, given the feedback) are set to -inf so they are not considered
    # one can then apply softmax over logits to get a probability distribution to either sample from or take the argmax of, depending on whether in training or eval
    def forward(self, h, valid_indices): # valid_indices is a list of indices for the remaining candidate words in the word list
        logits = self.policy(h) # thru linear layer
        
        # set invalid logits to -inf so softmax produces 0 there
        mask = torch.full_like(logits, float('-inf')).to(self.device)
        mask[valid_indices] = 0.0
        masked_logits = logits + mask
        
        return masked_logits # [vocab_size]