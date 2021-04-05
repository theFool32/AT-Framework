#!/usr/bin/env python3

from sys import implementation
from .loss import *


def get_loss_fn(name):
    if name == "CE":
        return cross_entropy
    else:
        raise NotImplementedError(f"Loss function not implemented: {name}")
