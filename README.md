# Stochastic Depth with PyTorch Hooks ![Travis CI build status](https://travis-ci.com/tasptz/pytorch-stochastic-depth.svg?branch=master) [![PyPI version](https://badge.fury.io/py/stochdepth.svg)](https://badge.fury.io/py/stochdepth)
A simple [hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) based implementation of [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382) for [torchvision resnets](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html).
# Example
```python
import torch
import torchvision.models as models
resnet = models.resnet152(pretrained=False)

from sdepth import set_hooks
set_hooks(resnet, p=0.2)

x = torch.zeros((8, 3, 224, 224), dtype=torch.float32, device=device)

resnet.train()
y = resnet(x)
```
