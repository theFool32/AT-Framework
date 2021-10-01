#!/usr/bin/env python3

import torch
from torch.nn import functional as F

from .base import Defense

from losses import get_loss_fn
from utils import Configurator

class AT(Defense):
    configuration = {
        "inner_loss": "CE",
        "outer_loss": "CE",
    }

    def __init__(self, _model, _attack):
        super(AT, self).__init__(_model, _attack)
        self.inner_loss_fn = get_loss_fn(Configurator().inner_loss)
        self.outer_loss_fn = get_loss_fn(Configurator().outer_loss)
        self.init_mode = "pgd"
        if Configurator().defense == "trades":
            self.init_mode = "trades"

    def train(self, data, label):
        output = self.model(data)
        loss = F.cross_entropy(output, label)

        is_model_training = self.model.training
        self.model.eval()
        adv_data = self.attack.perturb(
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
