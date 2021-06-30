import torch
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck

from stochdepth import set_hooks

def _input():
    return torch.zeros((4, 64, 8, 8), dtype=torch.float32).normal_()

def test_no_drop_basicblock_train():
    resnet = models.resnet18().train()
    basic_block = resnet.layer1[0]
    assert isinstance(basic_block, BasicBlock)
    x = _input()
    y = basic_block(x)
    set_hooks(resnet, p=0.)
    assert torch.allclose(y, basic_block(x))

def test_no_drop_basicblock_eval():
    resnet = models.resnet18().eval()
    basic_block = resnet.layer1[0]
    assert isinstance(basic_block, BasicBlock)
    x = _input()
    y = basic_block(x)
    set_hooks(resnet, p=0.)
    assert torch.allclose(y, basic_block(x))

def test_drop_basicblock_train():
    resnet = models.resnet18().train()
    basic_block = resnet.layer1[0]
    assert isinstance(basic_block, BasicBlock)
    x = _input()
    set_hooks(resnet, p=1.)
    assert torch.allclose(x, basic_block(x))

def test_drop_basicblock_eval():
    resnet = models.resnet18().eval()
    basic_block = resnet.layer1[0]
    assert isinstance(basic_block, BasicBlock)
    x = _input()
    y = basic_block(x)
    set_hooks(resnet, p=1.)
    assert torch.allclose(y, basic_block(x))

def test_no_drop_bottleneck_train():
    resnet = models.resnet50().train()
    bottleneck = resnet.layer1[0]
    assert isinstance(bottleneck, Bottleneck)
    x = _input()
    y = bottleneck(x)
    set_hooks(resnet, p=0.)
    assert torch.allclose(y, bottleneck(x))
    
def test_no_drop_bottleneck_eval():
    resnet = models.resnet50().eval()
    bottleneck = resnet.layer1[0]
    assert isinstance(bottleneck, Bottleneck)
    x = _input()
    y = bottleneck(x)
    set_hooks(resnet, p=0.)
    assert torch.allclose(y, bottleneck(x))

def test_drop_bottleneck_train():
    resnet = models.resnet50().train()
    bottleneck = resnet.layer1[0]
    assert isinstance(bottleneck, Bottleneck)
    x = _input()
    y = bottleneck.relu(bottleneck.downsample(x))
    set_hooks(resnet, p=1.)
    assert torch.allclose(y, bottleneck(x))

def test_drop_bottleneck_eval():
    resnet = models.resnet50().eval()
    bottleneck = resnet.layer1[0]
    assert isinstance(bottleneck, Bottleneck)
    x = _input()
    y = bottleneck(x)
    set_hooks(resnet, p=1.)
    assert torch.allclose(y, bottleneck(x))
