from copy import deepcopy
from collections.abc import Sequence
from typing import Callable

import torch
from torch import nn, Tensor
from torchvision.ops.misc import MLP
import einops

from eyemod.dl.models.resnet import ResNet18
from eyemod.dl.models.vit import AggregationCallback
from eyemod.dl.modules.embedding import TimeEmbedding, LearnedPositionalEmbedding


class SimpleFunctionPredictor(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module = ResNet18(pretrained=True, remove_linear=True),
        head: nn.Module = MLP(in_channels = 512, hidden_channels=[512, 256, 1]),
        embedding_dim = 512,
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.head = head
        self.pos_embedding = LearnedPositionalEmbedding(embedding_dim, 400)

    def forward(self, images: Tensor):

        z = self.forward_img(images=images)      

        if z.dim() == 4:
            # re-arrange the spatial tokens of the images to a single sequence of tokens
            z = einops.rearrange(z, 'b c sp0 sp1 -> b (sp0 sp1) c')

        y = self.head(z)

        return y
    
    def forward_img(self, images: Tensor) -> Tensor:

        B, *img_shape = images.shape
     
        z = self.img_encoder(images)
        
        if z.dim() == 3:
            return z
        elif z.dim() == 4:
            return self._add_positional_embedding(z)
        else:
            raise ValueError(f"Expected tensor with 3 or 5 dimensions, but got {z.ndim} dimensions")
    
    def _add_positional_embedding(self, x: Tensor) -> Tensor:
        """Add positional embedding to the feature maps such that they can be processed by a transformer"""
        
        B, C, sp0, sp1 = x.shape

        z = einops.rearrange(x, 'b c sp0 sp1 -> b (sp0 sp1) c')
        z = self.pos_embedding(z)
        z = einops.rearrange(z, 'b (sp0 sp1) c -> b c sp0 sp1', b=B, sp0=sp0, sp1=sp1)
    
        return z

class TemporalFunctionPredictor(nn.Module):

    def __init__(
        self,
        img_encoder: nn.Module = ResNet18(pretrained=True, remove_linear=True),
        time_encoder: nn.Module = TimeEmbedding(embedding_dim=512),
        head: nn.Module = MLP(in_channels = 3 * 512, hidden_channels=[512, 256, 1]),
        embedding_dim = 512,
        time_integration: str = 'add'
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.time_encoder = time_encoder
        self.head = head
        self.embedding_dim = embedding_dim
        self.integration_fn = get_time_integration_fn(time_integration)

        self.pos_embedding = LearnedPositionalEmbedding(embedding_dim, 400)

    def forward(self, img_sequence: Tensor, time_sequence: Tensor = None):
        # image_sequence shape [B, Seq, C, H, W]
        # time_sequence shape [B, S]

        z_img = self.forward_img(images=img_sequence)

        if time_sequence is not None and self.time_encoder and self.integration_fn:
            z_time = self.forward_time(times=time_sequence)
            z = self.integration_fn(z_img=z_img, z_time=z_time)
        else: 
            z = z_img            

        if z.dim() == 5:
            # re-arrange the spatial tokens of the images to a single sequence of tokens
            z = einops.rearrange(z, 'b s c sp0 sp1 -> b (s sp0 sp1) c')

        y = self.head(z)

        return y
    
    def forward_img(self, images: Tensor) -> Tensor:

        B, S, *img_shape = images.shape
        images = einops.rearrange(images, 'b s ... -> (b s) ...')

        z = self.img_encoder(images)
        z = einops.rearrange(z, '(b s) ... -> b s ...', b=B, s=S)
        
        if z.dim() == 3:
            return z
        elif z.dim() == 5:
            return self._add_positional_embedding(z)
        else:
            raise ValueError(f"Expected tensor with 3 or 5 dimensions, but got {z.ndim} dimensions")
    
    def forward_time(self, times: Tensor) -> Tensor:
        B, S, *_ = times.shape
        times = einops.rearrange(times, 'b s ... -> (b s) ...')
        z_time = self.time_encoder(times)
        z_time = einops.rearrange(z_time, '(b s) ... -> b s ...', b=B, s=S)
        return z_time

    def _add_positional_embedding(self, x: Tensor) -> Tensor:
        """Add positional embedding to the feature maps such that they can be processed by a transformer"""
        
        B, S, C, sp0, sp1 = x.shape

        z = einops.rearrange(x, 'b s c sp0 sp1 -> (b s) (sp0 sp1) c')
        z = self.pos_embedding(z)
        z = einops.rearrange(z, '(b s) (sp0 sp1) c -> b s c sp0 sp1', b=B, s=S, sp0=sp0, sp1=sp1)
    
        return z

class ModalityFunctionPredictor(nn.Module):

    def __init__(
        self,
        img_encoder: nn.Module = ResNet18(pretrained=True, remove_linear=True),
        head: nn.Module = MLP(in_channels = 3 * 512, hidden_channels=[512, 256, 1]),
        n_modalities: int = 2
    ):
        super().__init__()
        if isinstance(img_encoder, Sequence):
            self.img_encoders = nn.ModuleList(img_encoder)
        else:
            self.img_encoders = nn.ModuleList(
                [deepcopy(img_encoder) for i in range(n_modalities)]
            )
        self.head = head

    def forward(self, image_modalities: list[Tensor]):

        # expected shape of image tensor [B, C, H, W]
        assert isinstance(image_modalities, list), 'Input must be a list of image tensors of the different modalities'
        assert len(image_modalities) == len(self.img_encoders), "Number of provided modalities does not match the number of image encoders"

        z_img = []
        for images, encoder in zip(image_modalities, self.img_encoders):
            z_img.append(encoder(images))

        z = torch.stack(z_img, dim=1) # shape [B, S, embed_dim]

        y = self.head(z)

        return y

class MMTFunctionPredictor(nn.Module):
    def __init__(
    self,
    img_encoder: nn.Module = ResNet18(pretrained=True, remove_linear=True),
    time_encoder: nn.Module = TimeEmbedding(embedding_dim=512),
    head: nn.Module = MLP(in_channels = 3 * 512, hidden_channels=[512, 256, 1]),
    n_modalities: int = 2,
    time_integration: str = 'concat',
    embedding_dim: int = 512
    ):
        super().__init__()
        self.img_encoders = self._create_image_encoder(img_encoder, n_modalities)
        self.time_encoder = time_encoder
        self.head = head

        self.embedding_dim = embedding_dim
        self.integration_fn = get_time_integration_fn(time_integration)

    def _create_image_encoder(self, img_encoder: Sequence | nn.Module, n_modalities: int):
        if isinstance(img_encoder, Sequence):
            return nn.ModuleList(img_encoder)
        else:
            return nn.ModuleList([deepcopy(img_encoder) for i in range(n_modalities)])

    def forward(self, image_modalities: list[Tensor], time_sequence: Tensor = None):

        # expected shape of image tensor [B, Seq, C, H, W]
        # time_sequence shape [B, S]

        assert isinstance(image_modalities, list), 'Input must be a list of image tensors of the different modalities'
        assert len(image_modalities) == len(self.img_encoders), "Number of provided modalities does not match the number of image encoders"

        z_images = self._forward_images(image_modalities)

        if isinstance(self.integration_fn, Callable):
            z_time = self._forward_time(time_sequence)
            fn = self.integration_fn
            z_images = [fn(z, z_time) for z in z_images]

        z = torch.concat(z_images, dim=1) # shape [B, S, embed_dim]

        y = self.head(z)

        return y
    
    def _forward_images(self, image_modalities: list[Tensor]):
        # expected shape of image tensor [B, Seq, C, H, W]
        assert isinstance(image_modalities, list), 'Input must be a list of image tensors of the different modalities'
        assert len(image_modalities) == len(self.img_encoders), "Number of provided modalities does not match the number of image encoders"

        z_images = []
        for images, encoder in zip(image_modalities, self.img_encoders):
            B, S, *img_shape = images.shape

            #reshape from [B, S, C, H, W] to [(B * S), C, H, W] to process images as single batch
            img_reshape = images.reshape((B * S), *img_shape)
            z_img = encoder(img_reshape)
            z_img = z_img.reshape(B, S, self.embedding_dim)

            z_images.append(z_img)

        return z_images

    def _forward_time(self, time_sequence: Tensor = None):
            B, S, *_ = time_sequence.shape
            times = time_sequence.reshape((B * S), 1)
            z_t = self.time_encoder(times)                  # shape [B, embed_dim]
            z_t = z_t.reshape(B, S, self.embedding_dim)     # shape [B, S, embed_dim]
            return z_t


class MLPHead(MLP):
    """
    Modified MLP to process sequences.
    """
    def __init__(self, embedding_dim, num_tokens, hidden_channels, norm_layer = None, activation_layer = torch.nn.ReLU, inplace = None, bias = True, dropout = 0):
        self.embedding_dim = embedding_dim
        in_channels = embedding_dim * num_tokens
        super().__init__(in_channels, hidden_channels, norm_layer, activation_layer, inplace, bias, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Reshape the a sequence input to feed it to the MLP
        batch, seq, *_= x.shape
        x = torch.reshape(x, (batch, seq * self.embedding_dim))
        return super().forward(x)

class TransformerHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        dim_feedforward,
        num_heads,
        num_layers,
        aggregation_type = 'cls_token',
        num_classes = 1,
        dropout=0,
        norm=None,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        norm = nn.LayerNorm(embedding_dim)

        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers, norm=norm)
        self.class_token = nn.Parameter(torch.rand(1, 1, embedding_dim))
        self.aggregation_fn = AggregationCallback(agg_type=aggregation_type)
        self.head = MLP(in_channels=embedding_dim, hidden_channels=[num_classes])

    def forward(self, x: Tensor):
        
        # add class token
        n = x.shape[0]
        cls_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.encoder(x)
        x = self.aggregation_fn(x)
        x = self.head(x)
        return x


class VfTransformerHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        dim_feedforward,
        num_heads,
        num_layers,
        num_locations=59,
        dropout=0,
        norm=None,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.num_locations = num_locations

        norm = nn.LayerNorm(embedding_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer, num_layers=num_layers, norm=norm
        )
        self.location_token = nn.Parameter(torch.empty(1, num_locations, embedding_dim))
        nn.init.normal_(self.location_token)

        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, x: Tensor):

        # add class token
        n = x.shape[0]
        location_token = self.location_token.expand(n, -1, -1)
        x = torch.cat([location_token, x], dim=1)

        x = self.encoder(x)

        # extract location token
        x = x[:, 0:self.num_locations]
        x = self.linear(x)          # B, num_loc, 1 
        
        # remove trailing singleton dimension
        x = x.squeeze(dim=-1)       # B, num_loc, 1  -> B, num_loc

        return x


def get_time_integration_fn(integration_type: str):
    fn_dict = {
        'concat': _time_concat,
        'add': _time_addition,
        'none': None
    }
    return fn_dict[integration_type]

def _time_concat(z_img: Tensor, z_time: Tensor) -> Tensor:
    if z_img.dim() == 3:
        return torch.cat([z_img, z_time[:, 1:2, :]], dim=1)
    else:
        raise ValueError(f"Expected tensor with 3 dimensions, but got {z_img.dim()} dimensions")
    
def _time_addition(z_img: Tensor, z_time: Tensor) -> Tensor:
    if z_img.dim() == 3:
        return z_img + z_time
    elif z_img.dim() == 5:
        z_time = einops.rearrange(z_time, '... -> ... 1 1')
        z_time = z_time.expand_as(z_img)
        return z_img + z_time
    else:
        raise ValueError(f"Expected tensor with 3 or 5 dimensions, but got {z_img.dim()} dimensions")
