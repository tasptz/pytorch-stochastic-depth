import torch
from torch import nn
from torchvision.models import resnet
from typing import Callable, List

from .hook import register_hooks
from .iterator import resnet as resnet_iterator
from .target import get_targets

def func(modules: List[nn.Module], p: Callable[[float], float], targets=None):
    if targets is None:
        from .target import get_targets
        targets = get_targets()
    n = len(modules)
    return tuple(h for idx, m in enumerate(modules) for h in register_hooks(m, p(idx / (n - 1)), targets))

def resnet_linear(module: resnet.ResNet, max_p=0.5, targets=None):
    if targets is None:
        from .target import get_targets
        targets = get_targets()
    return func(resnet_iterator(module, tuple(targets.keys())), lambda x: x * max_p, targets)
