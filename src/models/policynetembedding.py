import torch
import torch.nn as nn

# given  a latent vector from the shared encoder, produced a predicted word embedding to compare with all word embeddings and produce a prob distribution
# this was chosen over a simpler output head producing logits over every word
# embeddings allow for a more finer-grained comparison between words
# word embeddings obtained by taking sum of letter embeddings (not counting feedback)
class PolicyHead(nn.Module):
    
    def __init__(self, hidden_dim, word_embed_dim, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.query_layer = nn.Linear(hidden_dim, word_embed_dim)

    def forward(self, h, valid_indices, word_embeddings):  
        # h: [batch_size, hidden_dim]
        query = self.query_layer(h)  # [batch_size, word_embed_dim]

        # Get candidate embeddings
        candidate_embeds = word_embeddings[valid_indices]  # [num_valid, word_embed_dim]

        # Compute scores via dot product
        scores = query @ candidate_embeds.T  # [batch_size, num_valid]

        return scores  # softmax to get prob distribution
