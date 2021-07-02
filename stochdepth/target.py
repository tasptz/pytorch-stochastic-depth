from torchvision.models import resnet

def get_targets():
    return {
        resnet.BasicBlock: (lambda x: x.bn2),
        resnet.Bottleneck: (lambda x: x.bn3)
    }
