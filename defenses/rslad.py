#!/usr/bin/env python3
# Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better

import sys

import torch
from torch.nn import functional as F

from .base import Defense
from models import get_network
from models.wideresnet import WideResNet34
from utils import Configurator
from losses import get_loss_fn

# Weights converted from wrn70-16
# weight = torch.load("./cifar10_linf_wrn70-16_with.pt")
# name_map = {
#     "init_conv.weight": "conv1.weight",
#     "batchnorm.bias": "bn1.bias",
#     "batchnorm.weight": "bn1.weight",
#     "batchnorm.running_mean": "bn1.running_mean",
#     "batchnorm.running_var": "bn1.running_var",
#     "batchnorm.num_batches_tracked": "bn1.num_batches_tracked",
#     "logits.bias": "fc.bias",
#     "logits.weight": "fc.weight",
# }
# layer_map = {
#     "batchnorm_0": "bn1",
#     "batchnorm_1": "bn2",
#     "conv_0": "conv1",
#     "conv_1": "conv2",
#     "shortcut": "convShortcut",
# }
# last_names = ["bias", "weight", "running_mean", "running_var", "num_batches_tracked"]
# for layer in range(11):
#     for block in range(4):
#         for k, v in layer_map.items():
#             for l in last_names:
#                 name_map[f"layer.{block}.block.{layer}.{k}.{l}"] = f"block{block+1}.layer.{layer}.{v}.{l}"

# new_weight = {}
# for k, v in weight.items():
#     new_weight[name_map[k]] = v
# torch.save(new_weight, "cifar10_linf_wrn70-16.pt")


def RSLAD_inner_loss(adv_logit, label, nat_logit=None, reduction="sum"):
    loss = F.kl_div(
        F.log_softmax(adv_logit, dim=1),
        F.softmax(label, dim=1),
        reduction="none",
    )
    if reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss.sum(1)


def RSLAD_outer_loss(adv_logit, label, nat_logit=None, reduction="mean"):
    kl_loss1 = F.kl_div(F.log_softmax(adv_logit, dim=1), F.softmax(label, dim=1))
    kl_loss2 = F.kl_div(F.log_softmax(nat_logit, dim=1), F.softmax(label, dim=1))
    loss = 5 / 6.0 * kl_loss1 + 1 / 6.0 * kl_loss2
    return loss


def AT_inner_loss(adv_logit, label, nat_logit=None, reduction="sum"):
    y = F.softmax(label, dim=1)
    loss = -y * F.log_softmax(adv_logit, dim=1)
    if reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss.sum(1)
    elif reduction == "mean":
        return loss.sum() / adv_logit.size(0)


def AT_outer_loss(adv_logit, label, nat_logit=None, reduction="mean"):
    return AT_inner_loss(adv_logit, label, nat_logit, reduction)


class RSLAD(Defense):
    # User should set teacher model by themself
    configuration = {
        "teacher_model": "WideResNet34",
        "teacher_model_weights": "",
        "teacher_model_mean": "0,0,0",
        "teacher_model_std": "1,1,1",
    }

    def __init__(self, _model, _attack):
        super(RSLAD, self).__init__(_model, _attack)
        self.inner_loss_fn = RSLAD_inner_loss
        self.outer_loss_fn = RSLAD_outer_loss
        self.init_mode = "trades"

        self.test_inner_loss_fn = get_loss_fn("CE")
        self.test_outer_loss_fn = get_loss_fn("CE")

        assert Configurator().teacher_model != ""
        assert Configurator().teacher_model_weights != ""

        teacher_mean = [float(v) for v in Configurator().teacher_model_mean.split(",")]
        teacher_std = [float(v) for v in Configurator().teacher_model_std.split(",")]
        teacher_mean = torch.tensor(teacher_mean).view(3, 1, 1).cuda()
        teacher_std = torch.tensor(teacher_std).view(3, 1, 1).cuda()

        self.teacher_model = get_network(Configurator().teacher_model, teacher_mean, teacher_std)
        # HACK: load from the author
        self.teacher_model.basic_net.module.load_state_dict(
            torch.load(Configurator().teacher_model_weights), strict=False
        )
        self.teacher_model.eval()

    def train(self, data, label):
        with torch.no_grad():
            teacher_label = self.teacher_model(data).detach()

        adv_data = self.attack.perturb(
            data, teacher_label, loss_fn=self.inner_loss_fn, init_mode=self.init_mode
        ).detach()
        adv_output = self.model(adv_data)

        output = self.model(data)
        loss = F.cross_entropy(output, label)

        adv_loss = self.outer_loss_fn(adv_output, teacher_label, output)

        total_loss = adv_loss

        return (
            output.detach(),
            adv_output.detach(),
            loss.item(),
            adv_loss.item(),
            total_loss,
        )

    def test(self, data, label, test_attack):
        adv_data = test_attack.perturb(
            data, label, loss_fn=self.test_inner_loss_fn, init_mode=self.init_mode
        ).detach()
        adv_output = self.model(adv_data)

        output = self.model(data)
        loss = F.cross_entropy(output, label)

        adv_loss = self.test_outer_loss_fn(adv_output, label, output)

        total_loss = adv_loss

        return (
            output.detach(),
            adv_output.detach(),
            loss.item(),
            adv_loss.item(),
            total_loss,
        )
