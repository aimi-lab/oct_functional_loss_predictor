from collections.abc import Iterable
from functools import partial

import torch
from torch import nn, Tensor
from torchvision.transforms import v2

from eyemod.dl.data.functionals import autocontrast

# class SampleTransform(nn.Module):
#     """
#     The base transform for all transformations.
#     The transform transforms only the entries specified by the key or keys.

#     Attributes:
#         key (str or tuple): The key or keys of the sample to transform.
#     """
#     def __init__(self, key: str | tuple) -> None:
#         """
#         Initialize the transform.

#         Args:
#             key (str or tuple): The key or keys of the sample to transform.
#         """
#         super().__init__()
#         if not isinstance(key, tuple):
#             key = (key,)
#         self.keys = key

#     def __call__(self, sample: dict):
#         return self.forward(sample)

#     def forward(self, sample: dict):
#         for key in self.keys:
#             sample[key] = self._transform(sample[key], key)
#         return sample

#     def _transform(self, x, key):
#         raise NotImplementedError
    
# class TransformWrapper(SampleTransform):
#     def __init__(self, key, transform):
#         super().__init__(key)
#         self.transform = transform

#     def _transform(self, x, key):
#       return self.transform(x)



class MultiChannel(nn.Module):
    """
    Generate a multi-channel image from a single channel image.

    Attributes:
        key (str or tuple): The key or keys of the sample to transform.
        n_channels (int): number of channels the output image should have.
    """
    def __init__(self,n_channels: int):
        super().__init__()
        self.n_channels = n_channels
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Expand the input tensor to the desired number of channels.
        Args:
            x: input tensor of shape (B, C, H, W) or (C, H, W)
        """
        if x.dim() == 3:
           return x.expand(self.n_channels, -1, -1)
        elif x.dim() == 4:
            return x.expand(-1, self.n_channels, -1, -1)

class MinMaxScaling(nn.Module):
    """
    Scales the values in a tensor from an old range to a new range.

    Attributes:
        old_range (tuple): The original range of the values.
        new_range (tuple): The desired range of the values.
    """
    def __init__(self, old_range: tuple, new_range: tuple,):
       super().__init__()
       self.old_min, self.old_max = old_range
       self.old_delta = self.old_max - self.old_min

       self.new_min, self.new_max = new_range
       self.new_delta = self.new_max - self.old_min

       assert self.old_delta > 0, "Invalid old_range, leads to zero division."

    def forward(self, x: Tensor):
        x_new = (x - self.old_min)/self.old_delta * self.new_delta + self.new_min
        return x_new
    
class AutoContrast(nn.Module):
    """
    Adjusts the contrast of an image by rescaling its pixel values.

    For more details see 'functionals.autocontrast'
    """
    def __init__(self, cutoff: tuple[float, float] = (0, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff = cutoff
        
    def forward(self, x: Tensor) -> Tensor:
        return autocontrast(x, self.cutoff)

class RandomAutoContrast(nn.Module):
    """
    Randomly applies the AutoContrast transformation to an image with a given probability.

    Attributes:
        p (float): Probability of applying the AutoContrast transformation.
        cutoff (tuple[float, float]): Cutoff values for the AutoContrast transformation.
    """
    def __init__(self, p: float = 0.5, cutoff: tuple[float, float] = (0, 1)):
        super().__init__()
        self.p = p
        self.autocontrast = AutoContrast(cutoff=cutoff)

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            return self.autocontrast(x)
        return x

class RandomGaussianNoise(nn.Module):
    def __init__(self, mean: float | tuple = 0, sigma: float | tuple = 0.1, clip = True):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def forward(self, x: torch.Tensor):
        mean = self._get_random(*self.mean) if isinstance(self.mean, Iterable) else self.mean
        sigma = self._get_random(*self.sigma) if isinstance(self.sigma, Iterable) else self.sigma
        print(sigma)
        return v2.functional.gaussian_noise(x, mean=mean, sigma=sigma, clip=self.clip)
       
    def _get_random(self, min, max):
        return torch.rand(1).item() * (max - min) + min


class NormalizeTensor(nn.Module):
    """
    Normalizes all values of a tensor based on the mean and std.

    Differs from torchvision.transform.Normalize. The torchvision implementation 
    expects an tensor of shape (..., C, H, W) representing an image.
    
    This implementation accepts any kind of tensor shape.
    """
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std
        assert self.std > 0, 'Standard deviation is zero, leads to zero division.'

    def forward(self, x: Tensor):
        return (x - self.mean) / self.std
    
class SqueezeDimension(nn.Module):
    """
    Squeezes the defined dimension if its size is one.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x.squeeze(dim=self.dim)

class FlattenTensor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: Tensor):
        return x.flatten()
    
class ScalarScaling(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x: Tensor):
        return x.mul(self.factor)
    
class RandomColumnRoll(nn.Module):
    """
    A transformation that randomly rolls the columns of an image by a specified range of pixels.
    This transformation shifts the columns of the input tensor along the last dimension (assumed to be the width)
    by a random amount within the specified range. The amount of shift is determined randomly for each input.
    Args:
        shift (int): The maximum absolute value of the shift. The actual shift will be a random integer
            in the range [-shift, shift] (inclusive).
    """
    def __init__(self, shift: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_low = shift * -1
        self.shift_high = shift + 1 # add one as upper bound is not included 
    
    def forward(self, x: Tensor):
        rand_shift = int(torch.randint(low=self.shift_low, high=self.shift_high, size=(1, 1)).squeeze())
        return torch.roll(x, rand_shift, dims=-1)

class ToDtype(nn.Module):
    """
    A transformation to change the dtype of an input.

    Allows specification of dtype as string, in contrast to the ToDtype transform implementation
    of torchvision. This is required when specifying transforms in the yaml config file.
    """
    def __init__(self, dtype: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(dtype, str), "dtype must be passed as string. If you want \
            to pass a torch.dtype use the torchvision implementation of ToDtype"
        try:
            self.dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype '{dtype}'. Ensure it matches a valid torch dtype (e.g., 'float32', 'int64').")
        
    def forward(self, x: Tensor):
        return x.to(self.dtype)
    
class ChannelSelect(nn.Module):
    """
    A PyTorch module for selecting specific channels from a tensor.
    This transform allows you to extract a subset of channels from multi-channel data,
    such as selecting specific color channels from RGB images or specific feature
    channels from neural network outputs.
    Args:
        channels (int | list[int]): The channel index(es) to select. Can be a single
            integer for one channel or a list of integers for multiple channels.
        channel_dim (int, optional): The dimension along which to select channels.
            Defaults to 0.
    Returns:
        Tensor: A tensor containing only the selected channels along the specified
            dimension.
    """
    def __init__(self, channels: int | list[int], channel_dim: int = 1):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels]
        self.channels = torch.tensor(channels)
        self.channel_dim = 0

    def forward(self, x: Tensor) -> Tensor:
        return torch.index_select(x, dim=self.channel_dim, index=self.channels)


class Crop(nn.Module):
    def __init__(self, top: int, left: int, height: int, width: int):
        super().__init__()
        self.crop_fn = partial(v2.functional.crop, top=top, left=left, height=height, width=width)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.crop_fn(x)    


if __name__ == '__main__':
    pass