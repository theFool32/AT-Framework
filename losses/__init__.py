#!/usr/bin/env python3

from sys import implementation
from .loss import *


def get_loss_fn(name):
    if name == "CE":
        return cross_entropy
    elif name == "trades_inner":
        return trades_inner
    elif name == "trades_outer":
        return trades_outer
    elif name == "mart_outer":
        return mart_outer
    else:
        raise NotImplementedError(f"Loss function not implemented: {name}")
