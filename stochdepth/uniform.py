import torch
from torch import nn
from torchvision.models import resnet

from .hook import register_hooks
from .iterator import unordered

def uniform(module: nn.Module, p=0., targets=None):
    if targets is None:
        from .target import get_targets
        targets = get_targets()
    target_types = tuple(targets.keys())
    if isinstance(module, target_types):
        return register_hooks(module, p, targets)
    else:
        return tuple(h for m in unordered(module, target_types) for h in register_hooks(m, p, targets))
