#!/usr/bin/env python3

import torch
from torch.nn import functional as F
from .base import Defense


class AT(Defense):
    def __init__(self, model, attack, **kw):
        super(AT, self).__init__(model, attack, **kw)

    def train(self, data, label):
        output = self.model(data)
        loss = F.cross_entropy(output, label)

        # self.model.eval()
        adv_data = self.attack.perturb(data, label).detach()
        # self.model.train()
        adv_output = self.model(adv_data)
        adv_loss = F.cross_entropy(adv_output, label)

        total_loss = adv_loss

        return output.detach(), adv_output.detach(), loss.item(), adv_loss.item(), total_loss
