import torch
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck

from stochdepth import set_hooks

def test_no_drop_basicblock():
    resnet = models.resnet18()
    basic_block = resnet.layer1[0]
    assert isinstance(basic_block, BasicBlock)
    x = torch.zeros((4, 64, 8, 8), dtype=torch.float32).normal_()
    y = basic_block(x)
    set_hooks(resnet, p=0.)
    assert torch.allclose(y, basic_block(x))

def test_drop_basicblock():
    resnet = models.resnet18()
    basic_block = resnet.layer1[0]
    assert isinstance(basic_block, BasicBlock)
    x = torch.zeros((4, 64, 8, 8), dtype=torch.float32).normal_()
    set_hooks(resnet, p=1.)
    assert torch.allclose(x, basic_block(x))

def test_no_drop_bottleneck():
    resnet = models.resnet50()
    bottleneck = resnet.layer1[0]
    assert isinstance(bottleneck, Bottleneck)
    x = torch.zeros((4, 64, 8, 8), dtype=torch.float32).normal_()
    y = bottleneck(x)
    set_hooks(resnet, p=0.)
    assert torch.allclose(y, bottleneck(x))

def test_drop_bottleneck():
    resnet = models.resnet50()
    bottleneck = resnet.layer1[0]
    assert isinstance(bottleneck, Bottleneck)
    x = torch.zeros((4, 64, 8, 8), dtype=torch.float32).normal_()
    y = bottleneck.relu(bottleneck.downsample(x))
    set_hooks(resnet, p=1.)
    assert torch.allclose(y, bottleneck(x))
