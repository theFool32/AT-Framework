#!/usr/bin/env python3
# https://github.com/VITA-Group/Alleviate-Robust-Overfitting/blob/4066275ee3/utils.py

import copy
import torch
from .base import Defense

__all__ = ["SWA"]


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
    BatchNorm buffers update (if any).
    Performs 1 epochs to estimate buffers average using train dataset.
    :param loader: train dataset loader for buffers average estimation.
    :param model: model being update
    :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


class SWA(Defense):
    def __init__(self, _model, _attack, _args, inner_defense):
        super(SWA, self).__init__(_model, _attack, _args)
        self.defense = inner_defense
        self.proxy_model = copy.deepcopy(_model)
        self.swa_n = 0
        try:
            self.start_epoch = int(_args.lr_adjust.split(",")[0])
        except Exception:
            self.start_epoch = 0

    def train(self, data, label):
        return self.defense.train(data, label)

    def test(self, data, label, test_attack=None):
        back = self.defense.test(data, label, test_attack)
        return back

    def postprocess(self, epoch):
        if epoch > self.start_epoch:
            # TODO: magic number 4
            # 2009.10526v1
            if epoch % 4 == 0 or epoch == self.args.max_epoch:
                moving_average(self.proxy_model, self.model, 1.0 / (self.swa_n + 1))
                self.swa_n += 1
                bn_update(self.args.dataset.train_loader, self.proxy_model)

                self.model.load_state_dict(self.proxy_model.state_dict())
