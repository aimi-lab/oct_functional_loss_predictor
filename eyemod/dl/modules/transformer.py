from copy import deepcopy

import torch
from torch import nn

# Useful sources:
# https://nn.labml.ai/transformers/index.html
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
# https://github.com/TheoPis/MOP/blob/color-fundus/models/ViT.py
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        # Attention
        self.norm_attn = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout, bias=bias
        )
        self.dropout = nn.Dropout(dropout)

        # Feed Forward
        self.norm_ff = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # deviation from the original implementation, normalize before
        z = self.norm_attn(x)
        self_attn = self.attention(query=z, key=z, value=z)
        self_attn = self.dropout(self_attn)
        x = x + self_attn

        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        ff = self.dropout(ff)
        x = x + ff

        return x


class FeedForward(nn.Sequential):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0, bias: bool = True):
        super().__init__()
        self.add_module(f"lin1", nn.Linear(embedding_dim, hidden_dim, bias=bias))
        self.add_module(f"relu1", nn.ReLU())
        self.add_module(f"drop1", nn.Dropout(dropout))
        self.add_module(f"lin2", nn.Linear(hidden_dim, embedding_dim, bias=bias))
        self.add_module(f"drop2", nn.Dropout(dropout))


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0, bias: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_dim = embedding_dim // num_heads

        head = AttentionHead(
            self.embedding_dim, self.attn_dim, dropout=dropout, bias=bias
        )
        self.heads = nn.ModuleList([deepcopy(head) for _ in range(num_heads)])
        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        attn = [head(query, key, value) for head in self.heads]
        attn = torch.concat(attn, dim=-1).contiguous()
        attn = self.W_o(attn)
        return attn


class AttentionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attn_dim: int = None,
        dropout: float = 0,
        bias: bool = True,
    ):
        super().__init__()
        if attn_dim is None:
            attn_dim = embedding_dim
        self.W_q = nn.Linear(embedding_dim, attn_dim, bias=bias)
        self.W_k = nn.Linear(embedding_dim, attn_dim, bias=bias)
        self.W_v = nn.Linear(embedding_dim, attn_dim, bias=bias)
        self.attention = Attention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        return self.attention(q, k, v, mask)


class Attention(nn.Module):
    def __init__(self, dropout: float = 0, mask_value: float = -1e9):
        super().__init__()
        self.mask_value = mask_value
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Scaled dot-product attention"""
        d_k = torch.tensor(query.shape[-1])
        attn = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)

        if mask is not None:
            attn.masked_fill(mask == 0, self.mask_value)

        attn = nn.functional.softmax(attn, dim=-1)

        attn = self.dropout(attn)

        attn = torch.matmul(attn, value)

        return attn
