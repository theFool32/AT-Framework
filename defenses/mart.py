#!/usr/bin/env python3

import torch
from .trades import TRADES

import sys

sys.path.insert(0, "..")
from losses import get_loss_fn


class MART(TRADES):
    def __init__(self, _model, _attack, _args):
        super(MART, self).__init__(_model, _attack, _args)
        self.inner_loss_fn = get_loss_fn("CE")
        self.outer_loss_fn = get_loss_fn("mart_outer")
