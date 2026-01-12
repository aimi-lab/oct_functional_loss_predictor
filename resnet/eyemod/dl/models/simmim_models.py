
import torch
from torch import nn

from .vit import VisionTransformer

class VitForSimMIM(VisionTransformer):
    def __init__(self, image_size, patch_size, in_channels, n_classes, num_layers, num_heads, hidden_dim, embedding_dim, aggregation_fn=..., dropout = 0):
        super().__init__(image_size, patch_size, in_channels, n_classes, num_layers, num_heads, hidden_dim, embedding_dim, aggregation_fn, dropout)

        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.encoder_stride = patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self._trunc_normal(self.mask_token, std=.02)

    def _trunc_normal(self, tensor, mean=0.0, std=1.0):
        nn.init.trunc_normal_(tensor, mean, std=std, a=-std, b=std)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        x = self.patch_embedding(x)

        assert mask is not None
        B, L, *_ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        # mask locations where mask is 1
        x = x * (1 - w) + mask_token * w

        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.pos_embedding(x)
        x = self.encoder(x)

        x = x[:, 1:]

        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x