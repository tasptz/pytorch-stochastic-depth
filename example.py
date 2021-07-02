import torch
import torchvision.models as models
resnet = models.resnet152(pretrained=False)
resnet.train()

from stochdepth import uniform
hooks = uniform(resnet, p=0.2)

x = torch.zeros((8, 3, 224, 224), dtype=torch.float32)
y = resnet(x)

# remove hooks
for h in hooks:
    h.remove()

from stochdepth import resnet_linear
hooks = resnet_linear(resnet)

y = resnet(x)
# remove hooks
for h in hooks:
    h.remove()