#!/usr/bin/env python3

from .base import Defense

class NoDefense(Defense):
    def __init__(self, model, attack, **kw):
        super(NoDefense, self).__init__(model, attack, **kw)

    def train(self, data, label):
        output = self.model(data)
        pred = torch.max(output, dim=1)[1]
        nat_acc = (pred == label).sum().item()
        loss = F.cross_entropy(output, label)

        total_loss = loss

        return output.detach(), None, loss.item(), 0, total_loss
