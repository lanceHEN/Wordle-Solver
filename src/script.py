from envs.WordleGame import WordleGame
from envs.wordleenv import WordleEnv
from models.letterencoder import LetterEncoder
from models.wordencoder import WordEncoder
from models.observationencoder import ObservationEncoder
from models.sharedencoder import SharedEncoder
from models.policynetembedding import PolicyHead
from models.valuenet import ValueHead
from training.trajectorycollector import generate_trajectory
from training.ppotrainer import ppo_update
from training.trainloop import training_loop
import torch

torch.autograd.set_detect_anomaly(True)

device = "cpu"

def load_word_list(path):
    with open(path, "r") as f:
        words = [line.strip() for line in f if line.strip()]
    return words



word_list = load_word_list('../data/5letterwords.txt')
answer_list = load_word_list('../data/5letterwords.txt')

env = WordleEnv(word_list=word_list, answer_list=answer_list)

env.reset()
obs, reward, done = env.step("slate")

#valid_indices = obs["valid_indices"]
#print(valid_indices)

le = LetterEncoder()
we = WordEncoder(le)
oe = ObservationEncoder(le)
#grid, meta = oe(obs)
#print(grid)
#print(meta)
se = SharedEncoder()
#h = se(grid.unsqueeze(0), meta.unsqueeze(0))
#print(h)

def generate_word_embeddings():
    return torch.stack([we(word) for word in word_list]).to(device)

word_embeddings = generate_word_embeddings()
ph = PolicyHead()
vh = ValueHead()

#logits = ph(h, [valid_indices], word_embeddings)
#value = vh(h)

#print(value)

shared_params = list(oe.parameters()) + list(se.parameters()) + list(we.parameters()) + list(le.parameters())
policy_params = shared_params + list(ph.parameters())  # Include policy head
value_params = shared_params + list(vh.parameters())  # Include value head

optimizer_policy = torch.optim.Adam(params=policy_params, lr=1e-3)
optimizer_value = torch.optim.Adam(params=value_params, lr=1e-3)
'''
traj = generate_trajectory(env, word_list, oe, se, ph, vh, word_embeddings)


ppo_update(oe, se, ph, vh, optimizer_policy, optimizer_value, traj["observations"],
           traj["actions"], traj["actions"], traj["returns"], traj["log_probs"],
           word_embeddings)

traj = generate_trajectory(env, word_list, oe, se, ph, vh, word_embeddings)

word_embeddings = generate_word_embeddings()

ppo_update(oe, se, ph, vh, optimizer_policy, optimizer_value, traj["observations"],
           traj["actions"], traj["actions"], traj["returns"], traj["log_probs"],
           word_embeddings)
'''
#print(word_embeddings.shape)
training_loop(WordleEnv(word_list, answer_list), oe, se, ph, vh, we, optimizer_policy, optimizer_value, word_list)