#!/usr/bin/env python3

import torch
from torch.nn import functional as F
from torch import nn

__all__ = ["cross_entropy", "trades_inner", "trades_outer", "mart_outer"]


def cross_entropy(adv_logit, label, nat_logit=None, reduction="mean"):
    return F.cross_entropy(adv_logit, label, reduction=reduction)


def trades_inner(adv_logit, label, nat_logit=None, reduction="sum"):
    loss = F.kl_div(
        F.log_softmax(adv_logit, dim=1),
        F.softmax(nat_logit, dim=1),
        reduction="none",
    )
    if reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss.sum(1)


def trades_outer(adv_logit, label, nat_logit=None, reduction="mean"):
    beta = 6  # TODO: adjustable
    nat_loss = F.cross_entropy(nat_logit, label)
    robust_loss = F.kl_div(
        F.log_softmax(adv_logit, dim=1),
        F.softmax(nat_logit, dim=1),
        reduction="batchmean",
    )

    return nat_loss + beta * robust_loss


def mart_outer(adv_logit, label, nat_logit=None, reduction="mean"):
    beta = 6
    adv_probs = F.softmax(adv_logit, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == label, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(adv_logit, label) + F.nll_loss(
        torch.log(1.0001 - adv_probs + 1e-12), new_y
    )

    nat_probs = F.softmax(nat_logit, dim=1)

    true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / adv_logit.size(0)) * torch.sum(
        torch.sum(
            F.kl_div(torch.log(adv_probs + 1e-12), nat_probs, reduction="none"), dim=1
        )
        * (1.0000001 - true_probs)
    )
    loss = loss_adv + float(beta) * loss_robust
    return loss
