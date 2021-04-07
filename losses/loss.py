#!/usr/bin/env python3

from torch.nn import functional as F

__all__ = ["cross_entropy"]


def cross_entropy(adv_logit, label, nat_logit=None, reduction="mean"):
    return F.cross_entropy(adv_logit, label, reduction=reduction)
