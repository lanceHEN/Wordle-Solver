import torch
import torch.nn as nn

# given a 3d tensor representing the game grid (word and feedback) of shape [max_guesses, word_length, embed_dim]
# and a 1d tensor representing the turn and number of candidates remaining
# produces a single vector with the given output dimension, output_dim
# this is a simple MLP (1 hidden layer) in practice, taking in the flatten grid concatenated with the 1d additional info tensor
class SharedEncoder(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim=256, output_dim=128, max_guesses=6, word_length=5):
        super().__init__()
        self.input_dim = max_guesses * word_length * embed_dim + 2
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, grid, meta):
        # grid: [max_guesses, word_length, embed_dim] -> flattened to max_guesses * word_length * embed_dim
        flat_grid = grid.view(-1)
        x = torch.cat([flat_grid, meta], dim=-1) # shape max_guesses * word_length * embed_dim + 2
        return self.encoder(x)