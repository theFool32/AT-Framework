#!/usr/bin/env python3

from torch.cuda.amp.autocast_mode import autocast
from torch import nn

from .resnet import *
from .preactresnet import *
from .wideresnet import *
from utils import Configurator


class ModelWrap(nn.Module):
    def __init__(self, basic_net, mean, std):
        super(ModelWrap, self).__init__()
        self.basic_net = nn.DataParallel(basic_net)
        self.mean = mean
        self.std = std
        self._test_mode = True

    def test_mode(self, to_test, no_amp=True):
        self._test_mode = to_test or no_amp
        return self._test_mode

    def forward(self, inputs):
        return self.basic_net((inputs - self.mean) / self.std)
        if self._test_mode:
            return self.basic_net((inputs - self.mean) / self.std)
        else:
            with autocast():
                return self.basic_net((inputs - self.mean) / self.std)


def get_network(model_name, mean=None, std=None):
    config = Configurator()
    if model_name == "PreActResNet18":
        model = PreActResNet18(num_classes=config.dataset.num_classes)
    elif model_name == "WideResNet28":
        model = WideResNet28(num_classes=config.dataset.num_classes)
    elif model_name == "WideResNet34":
        model = WideResNet34(num_classes=config.dataset.num_classes)
    else:
        raise NotImplementedError(f"Model not implemented: {model_name}")

    mean = mean if mean is not None else config.mean
    std = std if std is not None else config.std

    return ModelWrap(model.cuda(), mean, std).cuda()
