#!/usr/bin/env python3

import torch
from torch.nn import functional as F
import numpy as np

from .base import Defense
from losses import get_loss_fn
from utils import Configurator


def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.0
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1.0 - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, mult):
    return ramp_up(epoch, max_epochs, max_val, mult)


class TE(Defense):
    configuration = {
        "inner_loss": "CE",
        "outer_loss": "CE",
    }

    def __init__(self, _model, _attack):
        super(TE, self).__init__(_model, _attack)
        self.inner_loss_fn = get_loss_fn(Configurator().inner_loss)
        self.outer_loss_fn = get_loss_fn(Configurator().outer_loss)
        self.init_mode = "pgd"
        if Configurator().defense == "trades":
            self.init_mode = "trades"
        n_data = len(Configurator().dataset.train_loader.dataset)

        self.Z = torch.zeros(n_data, 10).cuda()  # intermediate values
        self.z = torch.zeros(n_data, 10).cuda()  # temporal outputs
        self.outputs = torch.zeros(n_data, 10).cuda()  # current outputs
        self.iter = 0
        self.batch_size = Configurator().batch_size
        self.w = 0

        self.sampler = Configurator().dataset.random_sampler

    def te_loss(self, z):
        def mse_loss(output):
            return F.mse_loss(z, output)
            # return F.mse_loss(F.softmax(output, dim=1), F.softmax(z, dim=1))

        def _loss_fn(loss_fn, adv_output, label, output, reduction=None):
            if reduction is None:
                loss_1 = loss_fn(adv_output, label, output)
            else:
                loss_1 = loss_fn(adv_output, label, output, reduction=reduction)
            return loss_1 + self.w * mse_loss(adv_output)

        def _inner_loss_fn(adv_output, label, output, reduction=None):
            return _loss_fn(self.inner_loss_fn, adv_output, label, output, reduction)

        def _outer_loss_fn(adv_output, label, output, reduction=None):
            return _loss_fn(self.outer_loss_fn, adv_output, label, output, reduction)

        return _inner_loss_fn, _outer_loss_fn

    def train(self, data, label):
        output = self.model(data)
        loss = F.cross_entropy(output, label)

        indices = self.sampler.indices[
            self.iter * self.batch_size : (self.iter + 1) * self.batch_size
        ]
        zcomp = self.z[indices]
        inner_loss_fn, outer_loss_fn = self.te_loss(zcomp)

        is_model_training = self.model.training
        self.model.eval()
        adv_data = self.attack.perturb(
            data, label, loss_fn=inner_loss_fn, init_mode=self.init_mode
        ).detach()
        if is_model_training:
            self.model.train()
        adv_output = self.model(adv_data)
        adv_loss = outer_loss_fn(adv_output, label, output)

        self.outputs[indices] = adv_output.data.clone().type_as(self.outputs)

        total_loss = adv_loss

        self.iter += 1

        return (
            output.detach(),
            adv_output.detach(),
            loss.item(),
            adv_loss.item(),
            total_loss,
        )

    def postprocess(self, epoch):
        alpha = 0.9
        self.w = weight_schedule(epoch, 200, 30, -5)
        self.Z = alpha * self.Z + (1.0 - alpha) * self.outputs
        self.z = self.Z * (1.0 / (1.0 - alpha ** (epoch + 1)))
        self.iter = 0

    def test(self, data, label, test_attack=None):
        output = self.model(data)
        loss = F.cross_entropy(output, label)

        is_model_training = self.model.training
        self.model.eval()
        adv_data = test_attack.perturb(
            data, label, loss_fn=self.inner_loss_fn, init_mode=self.init_mode
        ).detach()
        if is_model_training:
            self.model.train()
        adv_output = self.model(adv_data)
        adv_loss = self.outer_loss_fn(adv_output, label, output)

        total_loss = adv_loss

        return (
            output.detach(),
            adv_output.detach(),
            loss.item(),
            adv_loss.item(),
            total_loss,
        )
