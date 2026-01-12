from functools import partial
import torch

from lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from eyemod.dl.procedures._procedure_helper import unpack_metrics

class SupervisedTemporal(LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: partial[torch.optim.Optimizer] = partial(torch.optim.Adam, lr=1e-3),
        scheduler: partial[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        metrics: dict[str, Metric] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        if metrics is not None:
            self.train_metrics = MetricCollection(metrics, prefix="train_", postfix='_epoch')
            self.val_metrics = self.train_metrics.clone(prefix='val_', postfix='_epoch')
        else:
            self.train_metrics = None
            self.val_metrics = None

    def training_step(self, batch, batch_idx):

        inputs, time, targets, meta, *rest = batch

        assert isinstance(inputs, torch.Tensor), "First element of batch should be a tensor of images"
        assert isinstance(targets, torch.Tensor), "Second element of batch should be a tensor of targets"
        assert isinstance(meta, dict), "Meta data should be a dictionary"

        outputs = self.model(inputs, time)
        loss = self.criterion(outputs, targets)


        self.log(
            "loss",
            loss.item(),
            prog_bar=True,
            on_epoch=True,
            batch_size=len(inputs),
        )

        if self.train_metrics is not None:
            self.train_metrics.update(outputs, targets)

        return {"loss": loss, "output": outputs, "target": targets, "meta": meta} 

    def on_train_epoch_end(self):

        if self.train_metrics is not None:
            metrics = self.train_metrics.compute()
            metrics = unpack_metrics(metrics)
            self.log_dict(metrics, on_epoch=True, on_step=False)
            self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):

        inputs, time, targets, meta, *rest = batch

        assert isinstance(inputs, torch.Tensor), "First element of batch should be a tensor of images"
        assert isinstance(targets, torch.Tensor), "Second element of batch should be a tensor of targets"
        assert isinstance(meta, dict), "Meta data should be a dictionary"

        outputs = self.model(inputs, time)

        loss = self.criterion(outputs, targets)


        self.log(
            "val_loss",
            loss.item(),
            prog_bar=False,
            on_epoch=True,
            batch_size=len(inputs),
        )

        if self.val_metrics is not None:
            self.val_metrics.update(outputs, targets)

        return {"output": outputs, "target": targets, "meta": meta}

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
