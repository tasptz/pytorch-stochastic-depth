import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from typing import Mapping

def _forward_hook_scale(p):
    bernoulli = Bernoulli(torch.tensor(p, dtype=torch.float32))
    def f(module: nn.Module, _x, y):
        return y.mul_(1. - (bernoulli.sample().item() if module.training else p))
    return f

def register_hooks(module: nn.Module, p: float, targets: Mapping):
    return [targets[type(module)](module).register_forward_hook(_forward_hook_scale(p))]
