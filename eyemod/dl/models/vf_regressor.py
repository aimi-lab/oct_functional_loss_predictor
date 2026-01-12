import torch
from torch import nn

from torchvision.models import resnet50, ResNet50_Weights

class VFRegressor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights, progress=False)
        self.model.fc = nn.Linear(2048, 1)

        self.freeze_layers()
    
    def freeze_layers(self):

        for param in self.model.parameters():
            param.requires_grad = False

        layers = [self.model.layer4, self.model.fc]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.model(x)
