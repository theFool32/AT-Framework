#!/usr/bin/env python3

import torch
from torch.nn import functional as F
from torch import nn

__all__ = ["cross_entropy", "trades_inner", "trades_outer"]


def cross_entropy(adv_logit, label, nat_logit=None, reduction="mean"):
    return F.cross_entropy(adv_logit, label, reduction=reduction)


def trades_inner(adv_logit, label, nat_logit=None):
    return F.kl_div(
        F.log_softmax(adv_logit, dim=1), F.softmax(nat_logit, dim=1), reduction="sum"
    )


def trades_outer(adv_logit, label, nat_logit=None, reduction="mean"):
    beta = 6  # TODO: adjustable
    nat_loss = F.cross_entropy(nat_logit, label)
    robust_loss = F.kl_div(
        F.log_softmax(adv_logit, dim=1),
        F.softmax(nat_logit, dim=1),
        reduction="batchmean",
    )

    return nat_loss + beta * robust_loss
