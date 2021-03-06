#!/usr/bin/env python3

from .base import Attack

import torch
import sys

from losses import get_loss_fn
from utils import Configurator


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
):
    max_loss = torch.zeros(y.size(0), device=y.device)
    max_delta = torch.zeros_like(X)
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
        delta = torch.zeros_like(X)
        if init_mode == "pgd":
            if norm == "linf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l2":
                delta.normal_()
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r / (n+1e-10) * epsilon
        elif init_mode == "trades":
            delta = 0.001 * torch.randn_like(X).detach()
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
            if not Configurator().no_amp:
                # Configurator().scaler.scale(loss).backward()
                with Configurator().amp.scale_loss(loss, Configurator().opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "linf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l2":
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
    def __init__(self, model, epsilon=None, norm=None, alpha=None, iters=None):
        configurator = Configurator()
        self.epsilon = configurator.epsilon / 255 if epsilon is None else epsilon
        self.norm = configurator.norm if norm is None else norm
        self.alpha = configurator.pgd_alpha / 255 if alpha is None else alpha
        self.iters = configurator.attack_iters if iters is None else iters
        self.model = model

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
        )


class PGD_Test(Attack):
    """
    Used during training for test
    """

    def __init__(self, model, epsilon=None, norm=None, alpha=None, iters=None):
        configurator = Configurator()
        self.epsilon = configurator.epsilon / 255 if epsilon is None else epsilon
        self.norm = configurator.norm if norm is None else norm
        self.alpha = configurator.pgd_alpha / 255 if alpha is None else alpha
        self.iters = configurator.attack_iters if iters is None else iters
        self.model = model

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
        )
