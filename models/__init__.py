#!/usr/bin/env python3

from .resnet import *
from .preactresnet import *
from .wideresnet import *

from torch import nn


class ModelWrap(nn.Module):
    def __init__(self, basic_net, mean, std):
        super(ModelWrap, self).__init__()
        self.basic_net = basic_net
        self.mean = mean
        self.std = std

    def forward(self, inputs):
        return self.basic_net((inputs - self.mean) / self.std)


def get_network(args):
    model_name = args.model
    if model_name == 'PreActResNet18':
        model = PreActResNet18()
    elif model_name == 'WideResNet28':
        model = WideResNet28()
    elif model_name == 'WideResNet34':
        model = WideResNet34()
    else:
        raise NotImplementedError(f"Model not implemented: {model_name}")

    return ModelWrap(model, args.mean, args.std).cuda()
    # return ModelWrap(globals()[args.model](), args.mean, args.std).cuda()
