#!/usr/bin/env python3

import torch
from .at import AT

import sys

sys.path.insert(0, "..")
from losses import get_loss_fn


class TRADES(AT):
    def __init__(self, _model, _attack, **kw):
        super(TRADES, self).__init__(_model, _attack, **kw)
        self.inner_loss_fn = get_loss_fn("trades_inner")
        self.outer_loss_fn = get_loss_fn("trades_outer")
