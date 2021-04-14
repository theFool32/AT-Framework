#!/usr/bin/env python3

import torch
from torch.nn import functional as F
from .base import Defense


class NoDefense(Defense):
    def __init__(self, _model, _attack, _args):
        super(NoDefense, self).__init__(_model, _attack, _args)

    def train(self, data, label):
        output = self.model(data)
        # pred = torch.max(output, dim=1)[1]
        # nat_acc = (pred == label).sum().item()
        loss = F.cross_entropy(output, label)

        total_loss = loss

        return output.detach(), None, loss.item(), 0, total_loss
