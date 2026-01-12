import torch
from torch import Tensor

from torchvision.transforms._functional_tensor import _max_value, _assert_channels, _assert_image_tensor


def autocontrast(img: Tensor, cutoff: tuple[float, float] = (0, 1)) -> Tensor:
    """
    Adjusts the contrast of an image by rescaling its pixel values.
    This function rescales the pixel values of the input image such that the darkest pixel 
    becomes the minimum value of the image's data type (e.g., 0 for uint8), and the brightest 
    pixel becomes the maximum value of the data type (e.g., 255 for uint8). The `cutoff` 
    parameter allows for the exclusion of a specified lower and upper quantile of the pixel 
    value distribution, which can help in ignoring outliers.

    Args:
        img (Tensor): 
            The input image represented as a PyTorch tensor. The tensor should have a 
            shape of (..., C, H, W), where H is the height, W is the width, and C is the number of channels.
        cutoff (tuple[float, float]): 
            A tuple specifying the lower and upper quantile cutoff values. The first value 
            in the tuple represents the lower quantile (e.g., 0.02 for 2%), and the second 
            value represents the upper quantile (e.g., 0.98 for 98%). These values should 
            be in the range [0.0, 1.0].
    Returns:
        Tensor: 
            The contrast-adjusted image as a PyTorch tensor with the same shape as the input.
    Raises:
        ValueError: 
            If the `cutoff` values are not in the range [0.0, 1.0] or if the lower cutoff 
            is greater than or equal to the upper cutoff.
    Notes:
        Implementation is adapted from : https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py   
    """
    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    _assert_channels(img, [1, 3])

    if len(cutoff) != 2:
        raise ValueError(f"Exactly to cutoff values need to be provided, {len(cutoff)} where given") 
    
    if 0 > cutoff[0] or cutoff[1] > 1 or cutoff[0] >= cutoff[1]:
        raise ValueError(f"Cutoff values need to be in the range [0, 1] and lower must be smaller than upper value")

    cutoff = torch.Tensor(cutoff)

    bound = _max_value(img.dtype)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    # flatten img and cast to float to use quantile function
    img_flat = img.flatten(start_dim=-2).to(torch.float)
    minmax = img_flat.quantile(q=cutoff, dim=-1, interpolation='nearest', keepdim=True).to(dtype)
    minmax = minmax.unsqueeze(-1)

    minimum = minmax[0]
    maximum = minmax[1]

    scale = bound / (maximum - minimum)
    eq_idxs = torch.isfinite(scale).logical_not()
    minimum[eq_idxs] = 0
    scale[eq_idxs] = 1

    return ((img - minimum) * scale).clamp(0, bound).to(img.dtype)

