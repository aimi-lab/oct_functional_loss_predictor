from functools import partial
import torch

from lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from eyemod.dl.procedures._procedure_helper import unpack_metrics

class MeanPredictor(LightningModule):
    """
    General procedure which computes the mean target value of the train split and predicts this value in the val split.

    The predictor is designed as a direct substitute for a regular training procedure.    
    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        optimizer: partial[torch.optim.Optimizer] = None,
        scheduler: partial[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: torch.nn.Module = None,
        metrics: dict[str, Metric] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = criterion

        if metrics is not None:
            self.train_metrics = MetricCollection(metrics, prefix="train_", postfix='_epoch')
            self.val_metrics = self.train_metrics.clone(prefix='val_', postfix='_epoch')
        else:
            self.train_metrics = None
            self.val_metrics = None

        self.target_data = None
        self.target_mean = 0.0

    def training_step(self, batch, batch_idx):

        inputs, targets, *rest = batch

        assert isinstance(inputs, torch.Tensor), "First element of batch should be a tensor of images"
        assert isinstance(targets, torch.Tensor), "Second element of batch should be a tensor of targets"

        assert targets.shape[1] == 1, "Mean predictor only works for single value regression targets."
        
        if self.target_data is None:
            self.target_data = targets
        else:
            self.target_data = torch.concat((self.target_data, targets))
        
        step_mean = torch.mean(targets)
        outputs = torch.full_like(targets, step_mean)

        if self.criterion is None:
            loss = torch.tensor([0.0])
        else:
            loss = self.criterion(outputs, targets)
        self.log(
            "loss",
            loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(inputs),
        )

        if self.train_metrics is not None:
            self.train_metrics.update(outputs, targets)

        return {"loss": loss, "output": outputs, "target": targets} 
    
    def on_train_epoch_end(self):

        self.target_mean = torch.mean(self.target_data)
        self.target_data = None

        if self.train_metrics is not None:
            metrics = self.train_metrics.compute()
            metrics = unpack_metrics(metrics)
            self.log_dict(metrics, on_epoch=True, on_step=False)
            self.train_metrics.reset()     
        
    def backward(self, loss, *args, **kwargs):
        return

    def zero_grad(self, set_to_none = True):
        return

    def validation_step(self, batch, batch_idx):

        inputs, targets, *rest = batch

        assert isinstance(inputs, torch.Tensor), "First element of batch should be a tensor of images"
        assert isinstance(targets, torch.Tensor), "Second element of batch should be a tensor of targets"

        outputs = torch.full_like(targets, self.target_mean)

        if self.val_metrics:
            self.val_metrics.update(outputs, targets)

        return {"output": outputs, "target": targets}

    def on_validation_epoch_end(self):
        if self.val_metrics is not None:
            metrics = self.val_metrics.compute()
            metrics = unpack_metrics(metrics)
            self.log_dict(metrics, on_epoch=True, on_step=False)
            self.val_metrics.reset()

    def configure_optimizers(self):
        return