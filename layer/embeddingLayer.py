import torch as t
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(input_dim, output_dim)

    def forward(self, src):
        word_emb = self.embedding(src)
        return word_emb
