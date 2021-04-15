#!/usr/bin/env python3

from torch.cuda.amp.autocast_mode import autocast

from .resnet import *
from .preactresnet import *
from .wideresnet import *

from torch import nn


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
        if self._test_mode:
            return self.basic_net((inputs - self.mean) / self.std)
        else:
            with autocast():
                return self.basic_net((inputs - self.mean) / self.std)


def get_network(args):
    model_name = args.model
    if model_name == "PreActResNet18":
        model = PreActResNet18()
    elif model_name == "WideResNet28":
        model = WideResNet28()
    elif model_name == "WideResNet34":
        model = WideResNet34()
    else:
        raise NotImplementedError(f"Model not implemented: {model_name}")

    return ModelWrap(model.cuda(), args.mean, args.std).cuda()
    # return ModelWrap(globals()[args.model](), args.mean, args.std).cuda()
