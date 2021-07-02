from torch import nn
from torchvision.models.resnet import ResNet
from typing import Iterable

def unordered(module: nn.Module, target_types: Iterable):
    return tuple(m for m in module.modules() if isinstance(m, target_types))

def resnet(module: nn.Module, target_types: Iterable):
    targets = []
    for i in range(1, 5):
        layer = getattr(module, f'layer{i}')
        for t in layer:
            if isinstance(t, target_types):
                targets.append(t)
    return targets
