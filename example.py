import torch
import torchvision.models as models
resnet = models.resnet152(pretrained=False)

from stochdepth import register_forward_hooks
hooks = register_forward_hooks(resnet, p=0.2)

x = torch.zeros((8, 3, 224, 224), dtype=torch.float32)

resnet.train()
y = resnet(x)

# remove hooks
for h in hooks:
    h.remove()
