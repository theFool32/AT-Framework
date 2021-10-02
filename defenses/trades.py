#!/usr/bin/env python3

import torch

from .at import AT
from losses import get_loss_fn


class TRADES(AT):
    configuration = {
        "beta": 6,
        "inner_loss": "trades_inner",
        "outer_loss": "trades_outer",
    }

    def __init__(self, _model, _attack):
        super(TRADES, self).__init__(_model, _attack)
        self.init_mode = "trades"
