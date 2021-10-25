import numpy as np


class Piecewise_Lr_schedule:
    def __init__(self, opt, milestones, gamma=0.1):
        self.milestones = milestones
        self.gamma = gamma
        self.opt = opt
        self._step = 0

    def step(self, epoch=None):
        if epoch is not None:
            self._step = epoch
        else:
            self._step += 1

        if self._step in self.milestones:
            lr = self.opt.param_groups[0]["lr"]
            lr *= self.gamma
            self.opt.param_groups[0].update(lr=lr)


class Cosine_Lr_schedule:
    def __init__(self, opt, max_lr, max_epoch):
        self.opt = opt
        self.max_lr = max_lr
        self.max_epoch = max_epoch
        self._step = 0

    def step(self, epoch=None):
        if epoch is not None:
            self._step = epoch
        else:
            self._step += 1

        lr = self.max_lr * 0.5 * (1 + np.cos(self._step / self.max_epoch * np.pi))
        self.opt.param_groups[0].update(lr=lr)
