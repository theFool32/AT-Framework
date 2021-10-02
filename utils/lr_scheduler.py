
class Lr_schedule:
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
            lr = self.opt.param_groups[0]['lr']
            lr *= self.gamma
            self.opt.param_groups[0].update(lr=lr)
