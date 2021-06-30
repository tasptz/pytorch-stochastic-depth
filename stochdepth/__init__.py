import torch
from torch import nn
from torch.nn.functional import relu
from torch.distributions.bernoulli import Bernoulli
from torchvision.models import resnet

def _forward_hook(p):
    distribution = Bernoulli(torch.tensor(p, dtype=torch.float32))
    def f(module: nn.Module, x, _y):
        if module.training and distribution.sample():
            # drop
            return relu(*x) if module.downsample is None else relu(module.downsample(*x))
    return f

def register_forward_hooks(module: nn.Module, p=0., target_types=None) -> nn.Module:
    if target_types is None:
        target_types = resnet.BasicBlock, resnet.Bottleneck
    if isinstance(module, target_types):
        return [module.register_forward_hook(_forward_hook(p))]
    else:
        return [m.register_forward_hook(_forward_hook(p)) for m in module.modules() if isinstance(m, target_types)]

__all__ = [register_forward_hooks]
