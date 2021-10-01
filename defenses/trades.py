#!/usr/bin/env python3

import torch

from .at import AT
from losses import get_loss_fn


class TRADES(AT):
    configuration = {
        "beta": 6
    }

    def __init__(self, _model, _attack):
        super(TRADES, self).__init__(_model, _attack)
        self.inner_loss_fn = get_loss_fn("trades_inner")
        self.outer_loss_fn = get_loss_fn("trades_outer")
