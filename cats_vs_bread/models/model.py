import timm
import torch
import torch.nn as nn

from cats_vs_bread.configs import ModelConfig


class CatsVsBreadClassfier(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

        self.model = timm.create_model(
            self.model_config.model_name,
            pretrained=self.model_config.pretrained,
            num_classes=self.model_config.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
