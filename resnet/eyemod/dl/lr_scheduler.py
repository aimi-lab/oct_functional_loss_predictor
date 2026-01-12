from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR
from torch.optim import Optimizer


class WarmupLinearLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        else:
            return 1.0


class WarmupLinearCosineAnnealingLR(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        warmup_lr = WarmupLinearLR(optimizer, warmup_steps)
        cosine_lr = CosineAnnealingLR(
            optimizer, T_max=T_max - warmup_steps, eta_min=eta_min
        )
        super().__init__(optimizer, [warmup_lr, cosine_lr], [warmup_steps])
