import torch
from torch import nn
from torch.nn.functional import relu
from torch.distributions.bernoulli import Bernoulli
from typing import Mapping

def _forward_hook_skip(p):
    distribution = Bernoulli(torch.tensor(p, dtype=torch.float32))
    def f(module: nn.Module, x, _y):
        if module.training and distribution.sample():
            # drop
            return relu(*x) if module.downsample is None else relu(module.downsample(*x))
    return f

def _forward_hook_scale(p):
    def f(module: nn.Module, _x, y):
        if not module.training:
            return y * (1. - p)
    return f

def register_hooks(module: nn.Module, p: float, targets: Mapping):
    return module.register_forward_hook(_forward_hook_skip(p)), targets[type(module)](module).register_forward_hook(_forward_hook_scale(p))
