import torch
import torchmetrics


class VfMetricWrapper(torchmetrics.Metric):
    def __init__(self, metric: torchmetrics.Metric, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.ndim == 2, "Expected the prediction to have two dimensions"
        assert target.ndim == 2, "Expected the target to have two dimensions"

        preds = torch.mean(preds, dim=1)
        target = torch.mean(target, dim=1)
        self.metric.update(preds, target)

    def compute(self):
        return self.metric.compute()
    
class VfMeanAbsoluteError(VfMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric = torchmetrics.MeanAbsoluteError(), **kwargs)

class VfR2Score(VfMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(metric = torchmetrics.R2Score(), **kwargs)