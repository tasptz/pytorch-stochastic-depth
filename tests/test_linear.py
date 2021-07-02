import torch
import torchvision.models as models
import random
from torch.nn.functional import relu
from torchvision.models.resnet import BasicBlock, Bottleneck

from stochdepth import resnet_linear

def _input(channels=64):
    return torch.zeros((2, channels, 8, 8), dtype=torch.float32).normal_()

def _test_first(resnet, block_type):
    block = resnet.layer1[0]
    assert isinstance(block, block_type)
    x = _input()
    y = block(x)
    resnet_linear(resnet)
    assert torch.allclose(y, block(x))

def test_first_basic_train():
    resnet = models.resnet18().train()
    _test_first(resnet, BasicBlock)

def test_first_basic_eval():
    resnet = models.resnet18().eval()
    _test_first(resnet, BasicBlock)

def test_first_bottleneck_train():
    resnet = models.resnet50().train()
    _test_first(resnet, Bottleneck)

def test_first_bottleneck_eval():
    resnet = models.resnet50().eval()
    _test_first(resnet, Bottleneck)

def _test_last_train(resnet, block, x, f, p):
    f_x = f(x)
    identity = x if block.downsample is None else block.downsample(x)
    resnet_linear(resnet, p)
    y = block(x)
    assert torch.allclose(y, relu(f_x + identity)) or torch.allclose(y, relu(identity))

def test_last_basic_train():
    p = random.uniform(0.1, 0.9)
    resnet = models.resnet18().train()
    block = resnet.layer4[-1]
    assert isinstance(block, BasicBlock)
    x = _input(512)
    f = torch.nn.Sequential(
        block.conv1, block.bn1, block.relu, block.conv2, block.bn2
    )
    _test_last_train(resnet, block, x, f, p)

def test_last_bottleneck_train():
    p = random.uniform(0.1, 0.9)
    resnet = models.resnet50().train()
    block = resnet.layer4[-1]
    assert isinstance(block, Bottleneck)
    x = _input(2048)
    f = torch.nn.Sequential(
        block.conv1, block.bn1, block.relu, block.conv2, block.bn2, block.relu, block.conv3, block.bn3
    )
    _test_last_train(resnet, block, x, f, p)

def _test_last_eval(resnet, block, x, f, p):
    f_x = f(x)
    identity = x if block.downsample is None else block.downsample(x)
    resnet_linear(resnet, p)
    y = block(x)
    assert torch.allclose(y, relu(f_x * (1 - p) + identity))

def test_last_basic_train():
    p = random.uniform(0.1, 0.9)
    resnet = models.resnet18().eval()
    block = resnet.layer4[-1]
    assert isinstance(block, BasicBlock)
    x = _input(512)
    f = torch.nn.Sequential(
        block.conv1, block.bn1, block.relu, block.conv2, block.bn2
    )
    _test_last_eval(resnet, block, x, f, p)

def test_last_bottleneck_train():
    p = random.uniform(0.1, 0.9)
    resnet = models.resnet50().eval()
    block = resnet.layer4[-1]
    assert isinstance(block, Bottleneck)
    x = _input(2048)
    f = torch.nn.Sequential(
        block.conv1, block.bn1, block.relu, block.conv2, block.bn2, block.relu, block.conv3, block.bn3
    )
    _test_last_eval(resnet, block, x, f, p)
