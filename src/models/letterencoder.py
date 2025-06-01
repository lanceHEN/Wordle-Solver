import torch
import torch.nn as nn
import torch.nn.functional as F

# given a letter, produces rich embeddings with the given dimension
# this is useful for computing word embeddings for the policy network (take the sum of letter embeddings)
# as well as for encoding the state by taking embeddings for each guessed letter
class LetterEncoder(nn.Module):
    
    def __init__(self, letter_embed_dim=8, device=torch.device("cpu")):
        super().__init__()
        self.letter_embed_dim = letter_embed_dim
        self.device = device

        # letter embeddings (learnable)
        self.letter_embed = nn.Embedding(26, letter_embed_dim)

    # made in part with generative AI
    def forward(self, letter):
        letter_idx = ord(letter) - ord('a')
        letter_vec = self.letter_embed(torch.tensor(letter_idx, device=self.device))
        
        return letter_vec