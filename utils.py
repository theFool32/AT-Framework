#!/usr/bin/env python3


import os
import subprocess


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


class Parameters:
    def __init__(self, params):
        for k, v in params.items():
            self.__setattr__(k, v)

    def __setattr__(self, key, value):
        self.__dict__[key.lower()] = value

    def __repr__(self):
        return repr(self.__dict__)


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
