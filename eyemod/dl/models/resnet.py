import torch
from torch import nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    ResNet,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)
from torchvision.models.resnet import BasicBlock, Bottleneck

from .load_mixin import LoadMixin

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "ResNet50CIFAR"]


class ResNetWrapperBase(ResNet, LoadMixin):
    """
    Resnet wrapper class to allow instantiation of ResNet models with the model factory.
    """
    FREEZABLE_LAYERS = ["input_layer", "layer1", "layer2", "layer3", "layer4"]

    def __init__(
        self,
        model_getter: callable,
        weights=None,
        progress: bool = False,
        remove_linear: bool = False,
        remove_avg_pool: bool = False,
        in_channels: int = 3,
        out_features: int = 1000,
        layers_to_freeze: list[str] = None,
        additional_dropout: bool = False,
        remove_layer3: bool = False,
        remove_layer4: bool = False,
        **kwargs,
    ):
        model = model_getter(weights=weights, progress=False)
        self.__dict__.update(model.__dict__)

        self.remove_avg_pool = remove_avg_pool
        if remove_avg_pool:
            self.avgpool = nn.Identity()
            remove_linear = True

        if remove_linear:
            self.fc = nn.Identity()

        # dropout used by Davide Scandella
        self.additional_dropout = additional_dropout
        self.dropout = nn.Dropout(0.5)

        if remove_layer3:
            self.layer3 = nn.Identity()
        if remove_layer4:
            self.layer4 = nn.Identity()
        
        if out_features != 1000 and not remove_linear:
            if not remove_layer4:
                last_block = list(self.layer4.children())[-1]
            elif not remove_layer3:
                last_block = list(self.layer3.children())[-1]
            else:
                last_block = list(self.layer2.children())[-1]

            # depending on the resnet size the last block can be different
            # and with it the location of the batch norm
            if isinstance(last_block, BasicBlock):
                batch_norm = list(last_block.children())[-1]
            elif isinstance(last_block, Bottleneck):
                # last element is a ReLU, second last is batch_norm
                batch_norm = list(last_block.children())[-2]
            else:
                raise ValueError(f'Last block is of unknown type: {type(last_block)}')
            
            self.fc = nn.Linear(batch_norm.num_features, out_features)

        if in_channels != 3:
            # Change the first conv layer to accept different number of input channels
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if "checkpoint_path" in kwargs:
            self.from_checkpoint_path(**kwargs)

        if layers_to_freeze is not None:
            assert (
                weights is not None or "checkpoint_path" in kwargs
            ), "Model must be initialized with pretrained weights to freeze layers. Provide a checkpoint_path or pretrained weights."
            self.freeze_layers(layers_to_freeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.additional_dropout:
            x = self.dropout(x)

        if self.remove_avg_pool:
            return x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.additional_dropout:
            x = self.dropout(x)

        x = self.fc(x)

        return x

    def freeze_layers(self, layer_names: list[str]):
        """
        Freeze the weights of the specified layers.
        """
        assert all(
            [layer_name in self.FREEZABLE_LAYERS for layer_name in layer_names]
        ), f"Invalid layer name, must be one of {self.FREEZABLE_LAYERS}"

        if 'input_layer' in layer_names:
            layer_names.remove('input_layer')
            layer_names += ["conv1", "bn1"]

        # Convert to tuple for startswith function
        layer_names = tuple(layer_names)
        for name, param in self.named_parameters():
            if name.startswith(layer_names):
                param.requires_grad = False

class ResNet18(ResNetWrapperBase):
    def __init__(self,  pretrained: bool = False, progress: bool = False, **kwargs):
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        super().__init__(model_getter=resnet18, weights=weights, progress=False, **kwargs)


class ResNet34(ResNetWrapperBase):
    def __init__(self,  pretrained: bool = False, progress: bool = False, **kwargs):
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        super().__init__(model_getter=resnet34, weights=weights, progress=False, **kwargs)


class ResNet50(ResNetWrapperBase):
    def __init__(self, pretrained: bool = False, progress: bool = False, **kwargs):
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        super().__init__(model_getter=resnet50, weights=weights, progress=False, **kwargs)


class ResNet101(ResNetWrapperBase):
    def __init__(self,  pretrained: bool = False, progress: bool = False, **kwargs):
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        super().__init__(model_getter=resnet101, weights=weights, progress=False, **kwargs)


class ResNet152(ResNetWrapperBase):
    def __init__(self,  pretrained: bool = False, progress: bool = False, **kwargs):
        weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        super().__init__(model_getter=resnet152, weights=weights, progress=False, **kwargs)

class ResNet50CIFAR(ResNet50):
    """
    ResNet50 model for CIFAR-10 dataset according to the paper:
    A Simple Framework for Contrastive Learning of Visual Representations, Chen et al. 2020 
    """
    def __init__(self, weights=None, progress: bool = False, remove_linear: bool = True, **kwargs):
        super().__init__(weights=None, progress=False, remove_linear=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()

        # We have to load the weights after modifying the model, must not be done by super().__init__()
        if "checkpoint_path" in kwargs:
            self.from_checkpoint_path(**kwargs)

    def forward(self, x):
        return super().forward(x)
