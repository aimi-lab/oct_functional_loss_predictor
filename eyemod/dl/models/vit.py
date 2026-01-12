from copy import deepcopy

import torch
from torch import nn

from ..modules.transformer import EncoderLayer
from ..modules.embedding import PositionalEmbedding, PatchEmbedding
from .load_mixin import LoadMixin


class AggregationCallback():
    """
    Callback which wraps different aggregation functions.
    This is done to use hydras instantiate function to pass aggregation callback as
    a parameter to a model using the _target_ field.
    With Yaml and hydra it is not possible to pass a function callable, the object has to be instantiated.
    """
    def __init__(self, agg_type: str):
        if agg_type == 'max':
            self.agg_fn = max_aggregation_fn
        elif agg_type == 'mean':
            self.agg_fn = mean_aggregation_fn
        elif agg_type == 'cls_token':
            self.agg_fn = cls_token_aggregation_fn
        else:
            raise ValueError('Invalid aggregation type.')

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.agg_fn(x)

def max_aggregation_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Computes mean aggregation of sequence tokens

    Expected input of shape (batch, sequence, embedding)
    with class token at first position in the sequence.    
    """
    x = x[:, 1:] # remove class token
    x, _ = torch.max(x, dim=1)
    return x

def mean_aggregation_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Computes mean aggregation of sequence tokens

    Expected input of shape (batch, sequence, embedding)
    with class token at first position in the sequence.    
    """
    x = x[:, 1:] # remove class token
    x = torch.mean(x, dim=1)
    return x

def cls_token_aggregation_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Extracts class token from the sequence

    Expected input of shape (batch, sequence, embedding)
    with class token at first position in the sequence.    
    """
    return x[:, 0]

class VisionTransformer(nn.Module, LoadMixin):

    def __init__(
        self,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        n_classes: int, 
        num_layers,
        num_heads,
        hidden_dim,
        embedding_dim,
        aggregation_fn = AggregationCallback(agg_type='cls_token'),
        dropout: float = 0,
        **kwargs
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )

        num_tokens = 1 + self.patch_embedding.num_patches
        self.class_token = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.pos_embedding = PositionalEmbedding(embedding_dim=embedding_dim, num_tokens=num_tokens)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        norm = nn.LayerNorm(embedding_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers, norm=norm)
        self.aggregation_fn = aggregation_fn

        self.head = MLP(in_channels=embedding_dim, hidden_channels=[n_classes])

        if "checkpoint_path" in kwargs:
            self.from_checkpoint_path(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.patch_embedding(x)

        n = x.shape[0]

        cls_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.pos_embedding(x)
        x = self.encoder(x)

        x = self.aggregation_fn(x)

        x = self.head(x)

        return x


class Encoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        num_tokens: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(embedding_dim=embedding_dim, num_tokens=num_tokens)
        self.dropout = nn.Dropout(dropout)
        layer = EncoderLayer(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout, bias=bias)
        self.layers =  nn.Sequential(*[deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding(x)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.norm(x)
        return x


class MLP(nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: list[int], bias: bool = True, dropout: float = 0.0):
        """
        Args:
            in_channels (int): Number of input channels.
            hidden_channels (list[int]): List of hidden channels. Defines number of layers. Last entry represents output channels.
            bias (bool): Enable or disable the bias in the linear layers.
            dropout (float): Probability for the dropout layers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.dropout = dropout
        self._build()
        
    def _build(self):
        in_ = self.in_channels
        for layer, out_ in enumerate(self.hidden_channels):
            self.add_module(f'lin{layer}', nn.Linear(in_, out_, bias=self.bias))
            # Dont add activation and drop out at the end
            if layer < len(self.hidden_channels) - 1:
                self.add_module(f'relu{layer}', nn.ReLU())
                self.add_module(f'drop{layer}', nn.Dropout(self.dropout))
            in_ = out_





if __name__ == '__main__':
    pass
