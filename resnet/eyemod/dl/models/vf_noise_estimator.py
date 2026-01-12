import torch
from torch import nn
from torchvision.ops import MLP

from eyemod.dl.modules.embedding import _sinusoidal_embedding

class VfNoiseEstimator(nn.Module):
    def __init__(self, embedding_dim = 64, hidden_layers: list = [64, 1], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_dim = embedding_dim
        self.md_embedding = MdEmbedding(embedding_dim, base=10)
        self.estimator = MLP(in_channels=embedding_dim * 2, hidden_channels=hidden_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape

        x = torch.reshape(x, (b * n, 1))
        x_embed = self.md_embedding(x)
        x_embed = torch.reshape(x_embed, (b, n * self.embedding_dim))
        
        return self.estimator(x_embed)
    

class MdEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, base: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _sinusoidal_embedding(x, self.embedding_dim, self.base)
    
