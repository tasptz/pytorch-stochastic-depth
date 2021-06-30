import torch
from torch import nn
from torch.nn.functional import relu
from torch.distributions.bernoulli import Bernoulli
from torchvision.models import resnet

def _forward_hook_skip(p):
    distribution = Bernoulli(torch.tensor(p, dtype=torch.float32))
    def f(module: nn.Module, x, _y):
        if module.training and distribution.sample():
            # drop
            return relu(*x) if module.downsample is None else relu(module.downsample(*x))
    return f

def _forward_hook_scale(p):
    def f(module: nn.Module, _x, y):
        if module.eval:
            return y * (1. - p)
    return f

def register_forward_hooks(module: nn.Module, p=0., targets=None):
    if targets is None:
        targets = {
            resnet.BasicBlock: (lambda x: x.bn2),
            resnet.Bottleneck: (lambda x: x.bn3)
        }
    def f(m):
        return m.register_forward_hook(_forward_hook_skip(p)), targets[type(m)](m).register_forward_hook(_forward_hook_scale(p))
    target_types = tuple(targets.keys())
    if isinstance(module, target_types):
        return f(module)
    else:
        return tuple(h for m in module.modules() if isinstance(m, target_types) for h in f(m))

__all__ = [register_forward_hooks]
