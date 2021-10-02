#!/usr/bin/env python3

import torch
from .trades import TRADES

import sys

from losses import get_loss_fn


class MART(TRADES):
    configuration = {
        "beta": 6,
        "inner_loss": "CE",
        "outer_loss": "mart_outer",
    }

    def __init__(self, _model, _attack):
        super(MART, self).__init__(_model, _attack)
