#!/usr/bin/env python3

import torch
from torch.nn import functional as F
from .base import Defense

import sys
sys.path.insert(0, '..')
from losses import *

class AT(Defense):
    def __init__(self, model, attack, **kw):
        super(AT, self).__init__(model, attack, **kw)
        if 'inner_loss' in kw:
            self.inner_loss_fn = kw['inner_loss']
        else:
            self.inner_loss_fn = cross_entropy

        if 'outer_loss' in kw:
            self.outer_loss_fn = kw['outer_loss']
        else:
            self.outer_loss_fn = cross_entropy

    def train(self, data, label):
        output = self.model(data)
        loss = F.cross_entropy(output, label)

        # self.model.eval()
        adv_data = self.attack.perturb(data, label, self.inner_loss_fn).detach()
        # self.model.train()
        adv_output = self.model(adv_data)
        adv_loss = self.outer_loss_fn(adv_output, label, output)

        total_loss = adv_loss

        return output.detach(), adv_output.detach(), loss.item(), adv_loss.item(), total_loss
