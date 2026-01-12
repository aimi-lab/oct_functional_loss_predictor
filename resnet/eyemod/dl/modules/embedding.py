from collections.abc import Iterable

import torch
from torch import nn

class PatchEmbedding(torch.nn.Module):
    def __init__(self, image_size: int | Iterable[int, int], patch_size: int | Iterable[int, int], in_channels: int, embedding_dim: int):
        super().__init__()

        if isinstance(image_size, Iterable) and isinstance(patch_size, Iterable):
            assert all([i % p == 0 for i, p in zip(image_size, patch_size)]), 'Image is not divisible by patch without remainder'
            num_patches = (image_size[0] / patch_size[0]) * (image_size[1] / patch_size[1])
        elif isinstance(image_size, Iterable):
            assert all([i % patch_size == 0 for i in image_size]), 'Image is not divisible by patch without remainder'
            num_patches = (image_size[0] / patch_size) * (image_size[1] / patch_size)
        elif isinstance(patch_size, Iterable):
            assert all([image_size % p == 0 for p in patch_size]), 'Image is not divisible by patch without remainder'
            num_patches = (image_size / patch_size[0]) * (image_size / patch_size[1])
        else:
            assert image_size % patch_size == 0, 'Image is not divisible by patch without remainder'
            num_patches = pow(image_size / patch_size, 2)   
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = int(num_patches)
        
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            padding=0,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        # convert [..., embed_dim, patches] to [..., patches, embed_dim]
        x = torch.movedim(x, -1, -2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, num_tokens: int, learnable: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens

        if learnable:
            self.embedding = self._learnable()
        else:
            embedding = self._sinusoidal()
            self.register_buffer('embedding', embedding, persistent=True) 
      

    def _learnable(self):
        embedding = torch.rand(self.num_tokens, self.embedding_dim)
        return torch.nn.Parameter(embedding, requires_grad=True)

    def _sinusoidal(self):
        if self.embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even for sinusoidal embeddings, got {self.embedding_dim}")
        BASE = 10000
        positions = torch.arange(self.num_tokens)
        embedding = _sinusoidal_embedding(positions, self.embedding_dim, base=BASE)

        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)  # Get sequence length from input
        return x + self.embedding[:seq_len]
    
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_positions: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_positions, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        B, S, C, *_ = x.shape
        positions = torch.arange(S, device=x.device)        # S, 1
        positions = positions.unsqueeze(0).expand(B, -1)    # B, S, 1
        embeddings = self.embedding(positions)
        return x + embeddings

    
class TimeEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim: int, base: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _sinusoidal_embedding(x, self.embedding_dim, base=self.base)


def _sinusoidal_embedding(values: torch.Tensor, embedding_dim: int, base: int = 10000) -> torch.Tensor:
        """
        Computes sinusoidal positional embeddings for the values given in the input tensor.

        Args:
            values (torch.Tensor): Input tensor of shape (n_values,) or (n_values, 1) containing the values to embed.
            embedding_dim (int): The dimensionality of the embedding. Must be even
            base (int, optional): The base for the frequency calculation. Default is 10000.
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (n_values, embedding_dim).
        """
                
        if values.dim() ==  1:
            values = values.unsqueeze(-1)                   # shape (n_values, 1)
        values = values.expand((-1, embedding_dim))         # shape (n_values, embedding_dim)

        # tensor numerating the embedding dimensions
        dimensions = torch.arange(embedding_dim, device=values.device)[None, :]   # shape (1, embedding_dim)
        dimensions = dimensions.expand_as(values)           # shape (n_values, embedding_dim)

        # tensor containing the denominator (scaling) for each embedding index
        denominator = 2 * (dimensions // 2) / embedding_dim # shape (n_values, embedding_dim)
        denominator = torch.pow(base, denominator)          # shape (n_values, embedding_dim)

        angles = values / denominator                       # shape (n_values, embedding_dim)

        embedding = torch.zeros_like(angles, device=values.device)
        embedding[:, 0::2] = torch.sin(angles[:, 0::2])     # dim 2i
        embedding[:, 1::2] = torch.cos(angles[:, 1::2])     # dim 2i+1

        return embedding