from functools import partial

import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

from lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from eyemod.dl.procedures._procedure_helper import unpack_metrics

class SimMIM(LightningModule):
    """
    Implementation of the "SimMIM: a simple framework for masked image modeling" for pre-training
    vision transformers.

    Adapted from https://github.com/microsoft/SimMIM/
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: partial[optim.Optimizer] = partial(optim.Adam, lr=1e-3),
        scheduler: partial[optim.lr_scheduler._LRScheduler] = None,
        metrics: dict[str, Metric] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model

        self.mask_generator = MaskGenerator()
        self.img_channels = self.model.in_channels

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.model.embedding_dim,
                # this will only work with RGB images
                # TODO: find an alternative implementation. Class to get from patches to image
                out_channels=self.model.encoder_stride ** 2 * self.img_channels, kernel_size=1),
            nn.PixelShuffle(self.model.encoder_stride),
        )

        self.optimizer = optimizer
        self.scheduler = scheduler
        

        if metrics is not None:
            self.train_metrics = MetricCollection(metrics, prefix="train_", postfix='_epoch')
            self.val_metrics = self.train_metrics.clone(prefix='val_', postfix='_epoch')
        else:
            self.train_metrics = None
            self.val_metrics = None

    def training_step(self, batch, batch_idx):

        inputs, *rest = batch

        B, *_ = inputs.shape
        mask = self.mask_generator(batch_size=B)
        mask = mask.to(inputs.device)

        z = self.model(inputs, mask)

        rec = self.decoder(z)

        # inflate mask to image size
        mask = mask.repeat_interleave(self.model.patch_size, 1).repeat_interleave(self.model.patch_size, 2)
        # add channel dimension and make contiguous
        mask = mask.unsqueeze(1).contiguous()

        loss = masked_l1_loss(input=rec, target=inputs, mask=mask)

        # scale by number of channels
        loss = loss / self.model.in_channels

        self.log(
            "loss",
            loss.item(),
            prog_bar=True,
            on_epoch=True,
            batch_size=len(inputs),
        )

        if self.train_metrics is not None:
            self.train_metrics.update(rec, inputs)

        return {"loss": loss, "output": rec, "target": inputs, "mask": mask} 

    def on_train_epoch_end(self):

        if self.train_metrics is not None:
            metrics = self.train_metrics.compute()
            metrics = unpack_metrics(metrics)
            self.log_dict(metrics, on_epoch=True, on_step=False)
            self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):

        inputs, *rest = batch

        B, *_ = inputs.shape
        mask = self.mask_generator(batch_size=B, deterministic=True, seed=batch_idx)
        mask = mask.to(inputs.device)

        z = self.model(inputs, mask)

        rec = self.decoder(z)

        # inflate mask to image size
        mask = mask.repeat_interleave(self.model.patch_size, 1).repeat_interleave(self.model.patch_size, 2)
        # add channel dimension and make contiguous
        mask = mask.unsqueeze(1).contiguous()

        loss = masked_l1_loss(input=rec, target=inputs, mask=mask)

        # scale by number of channels
        loss = loss / self.model.in_channels

        self.log(
            "val_loss",
            loss.item(),
            prog_bar=False,
            on_epoch=True,
            batch_size=len(inputs),
        )

        if self.val_metrics is not None:
            self.val_metrics.update(rec, inputs)

        return {"output": rec, "target": inputs, "mask": mask}

    def on_validation_epoch_end(self):

        if self.val_metrics is not None:
            metrics = self.val_metrics.compute()
            metrics = unpack_metrics(metrics)
            self.log_dict(metrics, on_epoch=True, on_step=False)
            self.val_metrics.reset()

    def configure_optimizers(self):
        params = list(self.model.parameters())

        if self.scheduler:
            self.optimizer = self.optimizer(params)
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler(self.optimizer),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return self.optimizer(params)


class MaskGenerator:
    def __init__(self, input_size = 224, mask_patch_size= 32, model_patch_size=16, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self, batch_size: int = 1, deterministic: bool = False, seed: int = None):
        #TODO: change implementation to full torch
        
        if deterministic and seed is not None:
            # Save current state
            current_state = np.random.get_state()
            # Set deterministic seed
            np.random.seed(seed)
        
        masks = []
        for i in range(batch_size):
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
            mask = np.zeros(self.token_count, dtype=np.uint8)
            mask[mask_idx] = 1
            
            mask = mask.reshape((self.rand_size, self.rand_size))
            mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
            masks.append(mask)
        masks = np.stack(masks, axis=0)
        
        if deterministic and seed is not None:
            # Restore original state
            np.random.set_state(current_state)
            
        return torch.from_numpy(masks)


def masked_l1_loss(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert input.shape == target.shape, "Input and target must have same shape."
    assert input.shape[0] == mask.shape[0] and input.shape[2:] == mask.shape[2:], "Input and mask must have same shape except channel dimension."

    loss = F.l1_loss(input=input, target=target, reduction='none')
    loss = (loss * mask).sum() / (mask.sum() + eps)

    return loss
