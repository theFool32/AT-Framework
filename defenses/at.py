#!/usr/bin/env python3

import torch
from torch.nn import functional as F
from .base import Defense

import sys

sys.path.insert(0, "..")
from losses import get_loss_fn


class AT(Defense):
    def __init__(self, _model, _attack, **kw):
        super(AT, self).__init__(_model, _attack, **kw)
        self.inner_loss_fn = get_loss_fn(kw["inner_loss"])
        self.outer_loss_fn = get_loss_fn(kw["outer_loss"])
        self.init_mode = "pgd"
        if kw["defense"] == "trades":
            self.init_mode = "trades"

    def train(self, data, label):
        output = self.model(data)
        loss = F.cross_entropy(output, label)

        self.model.eval()
        adv_data = self.attack.perturb(
            data, label, loss_fn=self.inner_loss_fn, init_mode=self.init_mode
        ).detach()
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
