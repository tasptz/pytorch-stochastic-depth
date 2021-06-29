import torch
import torchvision.models as models
resnet = models.resnet152(pretrained=False)

from stochdepth import set_hooks
set_hooks(resnet, p=0.2)

x = torch.zeros((8, 3, 224, 224), dtype=torch.float32)

resnet.train()
y = resnet(x)