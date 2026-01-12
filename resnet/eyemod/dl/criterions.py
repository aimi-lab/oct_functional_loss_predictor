import torch
from torch import nn


class LinearWeightedMSELoss(nn.Module):
    """
    A weighted Mean Squared Error (MSE) loss module that computes MSE loss with weights
    derived from the target values.
    The weights are computed as a linear function of the target: `weights = target * slope + intercept`.
    This allows the loss to emphasize or de-emphasize different samples based on their target values.

    Args:
        slope (float): The slope coefficient for the linear weight function. Default is 1.
        intercept (float): The intercept coefficient for the linear weight function. Default is 1.

    """
    def __init__(self, slope: float = 1, intercept: float = 1):
        super().__init__()
        self._slope = slope
        self._intercept = intercept
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        weights = target * self._slope + self._intercept
        loss = nn.functional.mse_loss(input=input, target=target, reduction='none')
        loss = loss * weights
        return loss.mean()

    