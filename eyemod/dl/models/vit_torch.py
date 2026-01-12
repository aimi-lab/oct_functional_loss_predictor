import torch
import torch.nn as nn
from torchvision.models import vision_transformer
from eyemod.dl.models.vit import AggregationCallback


def get_vit_regressor(model_type: str, pretrained: bool = True) -> vision_transformer.VisionTransformer:

    if model_type == 'vit_b_16':
        getter = vision_transformer.vit_b_16
        weights = vision_transformer.ViT_B_16_Weights.DEFAULT

    elif model_type == 'vit_b_32':
        getter = vision_transformer.vit_b_32
        weights = vision_transformer.ViT_B_32_Weights.DEFAULT

    elif model_type == 'vit_l_16':
        getter = vision_transformer.vit_l_16
        weights = vision_transformer.ViT_L_16_Weights.DEFAULT

    elif model_type == 'vit_l_32':
        getter = vision_transformer.vit_l_32
        weights = vision_transformer.ViT_L_32_Weights.DEFAULT

    weights = weights if pretrained else None
    model = getter(weights = weights)

    new_head = nn.Linear(in_features=model.heads.head.in_features, out_features=1)
    model.heads.head = new_head

    return model


class VitWrapperBase(vision_transformer.VisionTransformer):

    def __init__(self, model_getter: callable, weights=None, num_classes: int = 1000, agg_type: str = 'cls_token'):

        model = model_getter(weights=weights, progress=False)
        self.__dict__.update(model.__dict__)

        self.aggregation_fn = AggregationCallback(agg_type=agg_type)

        if num_classes != 1000:
            new_head = nn.Linear(in_features=self.heads.head.in_features, out_features=1)
            self.heads.head = new_head

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = self.aggregation_fn(x)

        x = self.heads(x)

        return x
    
class ViT_B_16(VitWrapperBase):
    def __init__(self, pretrained: bool = False, num_classes = 1000, agg_type = 'cls_token'):
        weights = vision_transformer.ViT_B_16_Weights.DEFAULT if pretrained else None
        super().__init__(vision_transformer.vit_b_16, weights, num_classes, agg_type)

if __name__ == "__main__":
    model = get_vit_regressor("vit_b_16")
    print(model)
