#!/usr/bin/env python3
# https://github.com/csdongxian/AWP

from collections import OrderedDict
import copy
import sys

import torch
from torch.nn import functional as F

from .base import Defense

sys.path.insert(0, "..")
from losses import get_loss_fn

__all__ = ["AWP"]

EPS = 1e-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(
        model_state_dict.items(), proxy_state_dict.items()
    ):
        if len(old_w.size()) <= 1:
            continue
        if "weight" in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, loss_fn, gamma=0.01):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = copy.deepcopy(model)
        self.proxy_optim = torch.optim.SGD(self.proxy.parameters(), lr=0.01)
        self.gamma = gamma
        self.loss_fn = loss_fn

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss = -self.loss_fn(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class AWP(Defense):
    def __init__(self, _model, _attack, _args, inner_defense):
        super(AWP, self).__init__(_model, _attack, _args)
        try:
            loss_fn = inner_defense.inner_loss_fn
        except Exception:
            loss_fn = get_loss_fn("CE")

        self.awp = AdvWeightPerturb(_model, loss_fn)
        self.defense = inner_defense

    def train(self, data, label):

        is_model_training = self.model.training
        self.model.eval()
        adv_data = self.attack.perturb(
            data, label, loss_fn=self.defense.inner_loss_fn, init_mode=self.defense.init_mode
        ).detach()
        if is_model_training:
            self.model.train()

        diff = self.awp.calc_awp(adv_data, label)
        self.awp.perturb(diff)

        output = self.model(data)
        loss = F.cross_entropy(output, label)
        adv_output = self.model(adv_data)
        adv_loss = self.defense.outer_loss_fn(adv_output, label, output)

        total_loss = adv_loss
        self.args.opt.zero_grad()
        if not self.args.no_amp:
            # self.args.scaler.scale(total_loss).backward()
            # self.args.scaler.step(self.args.opt)
            # self.args.scaler.update()
            with self.args.amp.scale_loss(total_loss, self.args.opt) as scaled_loss:
                scaled_loss.backward()
            self.args.opt.step()
        else:
            total_loss.backward()
            self.args.opt.step()

        self.awp.restore(diff)


        _ = torch.zeros_like(total_loss)
        _.requires_grad_(True)

        return (
            output.detach(),
            adv_output.detach(),
            loss.item(),
            adv_loss.item(),
            total_loss.detach() + _,
        )

    def test(self, data, label, test_attack=None):
        return self.defense.test(data, label, test_attack)
