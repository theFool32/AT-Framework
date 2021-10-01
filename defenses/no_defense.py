#!/usr/bin/env python3

import torch
from torch.nn import functional as F
from .base import Defense


class NoDefense(Defense):
    def __init__(self, _model, _attack):
        super(NoDefense, self).__init__(_model, _attack)

    def train(self, data, label):
        output = self.model(data)
        loss = F.cross_entropy(output, label)

        total_loss = loss

        return output.detach(), None, loss.item(), 0, total_loss
