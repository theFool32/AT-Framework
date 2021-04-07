#!/usr/bin/env python3

from . import base

__all__ = ["cifar10_linf_AT_config", "cifar10_linf_TRADES_config"]

cifar10_linf_AT_config = {
    **base.linf_AT_config,
    "dataset": "cifar10",
}

cifar10_linf_TRADES_config = {
    **base.linf_TRADES_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_TRADES",
}
