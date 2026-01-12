
import torch
from torch import nn, Tensor
from torchvision.transforms import v2

import numpy as np


class GaussianNoise(nn.Module):
    """
    Add Gaussian noise to a tensor.

    Args:
        mean (float): Mean of the Gaussian noise. Defaults to 0.
        std (float): Standard deviation of the Gaussian noise. Defaults to 1.
        seed (int, optional): Random seed for reproducible noise generation. Defaults to None.
    """
    def __init__(self, mean: float = 0, std: float = 1):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add Gaussian noise to the input tensor.
        
        Args:
            x (Tensor): Input tensor of any shape.
            
        Returns:
            Tensor: Input tensor with added Gaussian noise.
        """
        noise = torch.normal(self.mean, self.std, size = x.shape, dtype=x.dtype)
        return x + noise
    
# TODO: Adapt implementation - Code was originally implemented by me in MOP repo of Theo
# class VisualFieldSmoothing(nn.Module):
#     """
#     Converts a measured (hard) visual field sensitivity value into a probability vector. The vector contains the probabilities of observing 
#     a given sensitivity value given the measured value.

#     params:
#         sensitivity_val_range: tuple, range of sensitivity values to consider. Default is (0, 40). Defines the length of the output probability vector.
#     """
#     def __init__(self, sensitivity_val_range: tuple = (0, 40)):
#         self.look_up_table = self._gen_lookup_table(sensitivity_val_range)
        
#     def _gen_lookup_table(self, sensitivity_val_range):

#         # Source of the formula:
#         # Henson et. al, “Response Variability in the Visual Field: Comparison of Optic Neuritis, 
#         # Glaucoma, Ocular Hypertension, and Normal Eyes,”
#         # Investigative Ophthalmology & Visual Science, vol. 41, no. 2, pp. 417–421, Feb. 2000.
#         SLOPE = -0.081
#         INTERCEPT = 3.27

#         sensitivity_vals = np.arange(*sensitivity_val_range)
#         sensitivity_vals = torch.linspace(*sensitivity_val_range, steps=1)
        
#         std = torch.exp(SLOPE * sensitivity_vals + INTERCEPT)
        
#         probability_LUT = np.zeros((len(sensitivity_vals), len(sensitivity_vals)))

#         for current_sensitivity in sensitivity_vals:
#             # generate a probability distribution for each sensitivity value parameterized by the standard deviation
#             probability_LUT[current_sensitivity] = norm.pdf(
#                 sensitivity_vals, loc=current_sensitivity, scale=std[current_sensitivity]
#             )
#         probability_LUT = probability_LUT.astype(np.float32)
        
#         return probability_LUT
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Source of the formula:
#         # Henson et. al, “Response Variability in the Visual Field: Comparison of Optic Neuritis, 
#         # Glaucoma, Ocular Hypertension, and Normal Eyes,”
#         # Investigative Ophthalmology & Visual Science, vol. 41, no. 2, pp. 417–421, Feb. 2000.
#         SLOPE = -0.081
#         INTERCEPT = 3.27

#         # scale the standard deviation based on the intensity value
#         std = torch.exp(SLOPE * x + INTERCEPT)

#         noise = torch.normal(0, std, size = x.shape, dtype=x.dtype)
#         return x + noise

#     def _md_to_sensitivity(md: float) -> float:
#         Slope: -1.0371, Intercept: 26.1770