import torch
import torch.nn as nn

# a critic network that takes in the vector for the current game from SharedEncoder and produces an estimate of the value for that state
# apply softmax over logits to get a probability distribution to either sample from or take the argmax of
# this is a simple MLP with one linear layer - single scalar output
class CriticHead(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        return self.value(h).item() # scalar