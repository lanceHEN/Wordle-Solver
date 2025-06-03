import torch
import torch.nn as nn

# given batched latent vectors from the shared encoder, produces predicted word embeddings, which are then
# compared with all word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
# which one can softmax over to get prob. dists.
# this was chosen over a simpler output head immediately producing logits over every word because embeddings allow for a finer-grained comparison between words
class PolicyHead(nn.Module):
    
    # Initializes a PolicyHead with the given input latent dimension (should be same as SharedEncoder output), word embedding dimension, and device
    def __init__(self, hidden_dim, word_embed_dim, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(hidden_dim, word_embed_dim) # linear

    # given batched latent vectors from the shared encoder, produces predicted word embeddings, which are then
    # compared with all word embeddings for all guess words (via dot prod.) to produce logits (masked for valid word indices),
    # which one can softmax over to get prob. dists.
    # this was chosen over a simpler output head immediately producing logits over every word because embeddings allow for a finer-grained comparison between words
    def forward(self, h, valid_indices_batch, word_embeddings):

        # h: [B, hidden_dim]
        query = self.linear(h)  # [B, word_embed_dim]

        # compute all scores: [B, vocab_size]
        scores = query @ word_embeddings.T  # [B, vocab_size]

        # mask out invalid logits (set to -inf)
        mask = torch.full_like(scores, float('-inf'))  # [B, vocab_size]
        for i, valid_idx in enumerate(valid_indices_batch):
            mask[i, valid_idx] = 0.0  # keep only valid words

        masked_logits = scores + mask  # [B, vocab_size]
        return masked_logits
