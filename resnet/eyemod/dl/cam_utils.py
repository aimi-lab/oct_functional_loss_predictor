from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import cv2

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

import torchcam.methods

import pytorch_grad_cam
import pytorch_grad_cam.base_cam as pgcb
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class CAMProtocol(Protocol):
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        ...

    def __call__(self, input: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        ...


class SimpleCAM():
    """
    Basic Class Activation Map (CAM) implementation. Only works with CNNs Global Average Pooling layer at the end.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.handels = []
        self.activations = None
        self.img_shape = None

        self.target_layer.register_forward_hook(self._store_activations)
        assert hasattr(self.model, 'fc'), 'Provided model is missing an fully connected (fc) layer'

    def _store_activations(self, module, args, output) -> None:
        self.activations = output.detach()
        return 

    def _compute_cam(self):
        activations = self.activations.movedim(1, -1)
        prediction_map = self.model.fc(activations)
        prediction_map = prediction_map.movedim(-1, 1)
        
        return prediction_map.detach().cpu()
    
    def __call__(self, input_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:

        img_shape = input_tensor.shape[-2:]

        output = self.model(input_tensor)
         
        cam = self._compute_cam()

        if output.shape[-1] > 1:
            cam = torch.nn.functional.relu(cam)

        cam_resized = _resize_cam(cam, img_shape=img_shape)
        # convert cam to the same shape and dtype as outputted by the python-grad-cam package
        # remove channel dimension
        cam_resized = cam_resized.squeeze(1)

        output = output.detach().cpu()

        return cam_resized, output

class TorchCAMWrapper():
    def __init__(self, model: nn.Module, cam_algorithm: torchcam.methods.core._CAM, class_idx: int =  0):
        assert isinstance(cam_algorithm, torchcam.methods.core._CAM), "Provided CAM Algorithm is not part of the torchcam package."
        self.model = model
        self.cam = cam_algorithm
        self.class_idx = class_idx

    def __call__(self, input_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        img_shape = input_tensor.shape[-2:]
    
        output = self.model(input_tensor)
  
        out_cam = self.cam(class_idx=self.class_idx, scores=output, normalized=False)
        out_cam = out_cam[0].cpu() # we only support CAM of one class, get corresponding map
        out_cam = _resize_cam(out_cam, img_shape=img_shape)
        out_cam = out_cam

        output = output.detach().cpu()

        return out_cam, output

class CAM_tc(TorchCAMWrapper):
    def __init__(self, model, target_layer, fc_layer, class_idx = 0):
        super().__init__(
            model=model, 
            cam_algorithm=torchcam.methods.CAM(model=model, target_layer=target_layer, fc_layer=fc_layer),
            class_idx=class_idx,
        )

class GradCAM_tc(TorchCAMWrapper):
    def __init__(self, model, target_layer, class_idx = 0):
        super().__init__(
            model=model,
            cam_algorithm=torchcam.methods.GradCAM(model=model, target_layer=target_layer,),
            class_idx=class_idx,
        )

class GradCAMpp_tc(TorchCAMWrapper):
    def __init__(self, model, target_layer, class_idx = 0):
        super().__init__(
            model=model,
            cam_algorithm=torchcam.methods.GradCAMpp(model=model, target_layer=target_layer),
            class_idx=class_idx,
        )

class PytorchGradCAMWrapper():
    def __init__(self, model: nn.Module, cam_algorithm: pytorch_grad_cam.base_cam.BaseCAM, class_idx: int =  0):
        assert isinstance(cam_algorithm, pytorch_grad_cam.base_cam.BaseCAM), "Provided CAM Algorithm is not part of the pytorch_grad_cam package."
        self.model = model
        self.cam = cam_algorithm
        self.targets = [ClassifierOutputTarget(class_idx)]

    def __call__(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_shape = input_tensor.shape[-2:]

        output = self.model(input_tensor)
        output = output.detach().cpu()
            
        out_cam = self.cam(input_tensor, targets=self.targets)
        out_cam = _resize_cam(out_cam, img_shape=img_shape)
        out_cam = out_cam

        return out_cam, output

class GradCAM_pgc(PytorchGradCAMWrapper):
    def __init__(self, model, target_layer, class_idx = 0):
        super().__init__(
            model=model,
            cam_algorithm=pytorch_grad_cam.GradCAM(model=model, target_layers=[target_layer]),
            class_idx=class_idx,
        )

class GradCAMpp_pgc(PytorchGradCAMWrapper):
    def __init__(self, model, target_layer, class_idx = 0):
        super().__init__(
            model,
            cam_algorithm=pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=[target_layer]),
            class_idx=class_idx,
        )

def _resize_cam(cam: torch.Tensor | np.ndarray, img_shape: tuple) -> torch.Tensor:
    if isinstance(cam, np.ndarray):
        cam = torch.from_numpy(cam)
    cam_resized = resize(cam, img_shape)
    return cam_resized

def visualize_cam(image: np.ndarray, cam: np.ndarray, norm_range: tuple = None, cmap = 'jet', alpha = 0.3) -> tuple[Figure, Axes]:
    """
    Visualize a Class Activation Map (CAM) overlaid on an image.
    Args:
        image (np.ndarray): The input image as a numpy array of shape CxHxW.
        cam (np.ndarray): The Class Activation Map as a numpy array CxHxW.
        norm_range (tuple, optional): A tuple (min, max) specifying the normalization range 
            for the CAM values. If None, uses the min and max values from the CAM. 
            Defaults to None.
        cmap (str, optional): The colormap name to use for visualizing the CAM. 
            Defaults to 'jet'.
    Returns:
        tuple[Figure, Axes]: A tuple containing the matplotlib Figure and Axes objects 
            with the CAM visualization overlaid on the image.
    """

    # move channel to last dim for visualization
    image = np.moveaxis(image, 0, -1)
    cam = np.moveaxis(cam, 0, -1)

    assert image.shape[:2] == cam.shape[:2], "Image and cam dont have same height and width."
    assert cam.shape[-1] == 1, "More than one CAM provided."

    fig, ax = plt.subplots(1, 1)

    img_cmap = 'gray' if image.shape[-1] == 1 else None
    ax.imshow(image, cmap=img_cmap)

    norm =  None if norm_range is None else mpl.colors.Normalize(vmin=norm_range[0], vmax=norm_range[1])
    cam_img = ax.imshow(cam, cmap=cmap,norm=norm, alpha=alpha)
    fig.colorbar(cam_img)

    ax.axis('off')

    return fig, ax

def visualize_cam_davide(image: np.ndarray, cam: np.ndarray, norm_range: tuple = None, cmap = 'jet', alpha = 0.2):
    
    image = np.moveaxis(image, 0, -1)
    cam = np.moveaxis(cam, 0, -1)

    assert image.shape[:2] == cam.shape[:2], "Image and cam dont have same height and width."
    assert cam.shape[-1] == 1, "More than one CAM provided."
   
    scaled_img = (image - image.min()) / (image.max() - image.min())
    bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)

    colormap_dict = {
    'jet': cv2.COLORMAP_JET,
    'hot': cv2.COLORMAP_HOT,
    'cool': cv2.COLORMAP_COOL,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'plasma': cv2.COLORMAP_PLASMA,
    'inferno': cv2.COLORMAP_INFERNO,
    'bone': cv2.COLORMAP_BONE,
    'rainbow': cv2.COLORMAP_RAINBOW,
    }
    colormap = colormap_dict[cmap]

    cam_img = show_cam_on_image(
            bgr_img,
            cam,
            use_rgb=False,
            colormap=colormap,
            image_weight=(1-alpha),
        )

    fig, ax = plt.subplots(1, 1)
    ax.imshow(cam_img )
    ax.axis('off')
    
    return fig, ax

