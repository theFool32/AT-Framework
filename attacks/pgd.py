#!/usr/bin/env python3

from torch.random import initial_seed
from .base import Attack

import torch
import sys

sys.path.insert(0, "..")
from losses import get_loss_fn


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(
    model,
    X,
    y,
    epsilon,
    alpha,
    attack_iters,
    restarts,
    norm,
    early_stop=False,
    loss_fn=None,
    init_mode="pgd",
    args=None,
):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    init_mode = init_mode.lower()
    if loss_fn is not None:
        with torch.no_grad():
            is_model_training = model.training
            model.eval()
            nat_output = model(X).detach()
            if is_model_training:
                model.train()
    else:
        nat_output = None
        loss_fn = get_loss_fn("CE")

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if init_mode == "pgd":
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r / n * epsilon
        elif init_mode == "trades":
            delta = 0.001 * torch.randn(X.shape).to(delta.device).detach()
        else:
            raise ValueError
        delta = clamp(delta, -X, 1 - X)
        delta.requires_grad = True

        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = loss_fn(output, y, nat_output)
            if not args.no_apex:
                with args.amp.scale_loss(loss, args.opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (
                    (d + scaled_g * alpha)
                    .view(d.size(0), -1)
                    .renorm(p=2, dim=0, maxnorm=epsilon)
                    .view_as(d)
                )
            d = clamp(d, -x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = loss_fn(model(X + delta), y, nat_output, reduction="none")
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


class PGD(Attack):
    def __init__(self, args, model):
        self.epsilon = args.epsilon / 255
        self.norm = args.norm
        self.alpha = args.pgd_alpha / 255
        self.iters = args.attack_iters
        self.model = model
        self.args = args

    def perturb(
        self, inputs, labels=None, loss_fn=None, early_stop=False, init_mode="pgd"
    ):
        return inputs + attack_pgd(
            self.model,
            inputs,
            labels,
            self.epsilon,
            self.alpha,
            self.iters,
            1,
            self.norm,
            early_stop=early_stop,
            loss_fn=loss_fn,
            init_mode=init_mode,
            args=self.args,
        )
