#!/usr/bin/env python3

from . import base

cifar100_linf_AT_config = {
    **base.linf_AT_config,
    "dataset": "cifar100",
    "fname": "cifar100_linf_AT",
}

cifar100_linf_TRADES_config = {
    **base.linf_TRADES_config,
    "dataset": "cifar100",
    "fname": "cifar100_linf_TRADES",
}

cifar100_linf_MART_config = {
    **base.linf_MART_config,
    "dataset": "cifar100",
    "fname": "cifar100_linf_MART",
}

cifar100_l2_AT_config = {
    **base.l2_AT_config,
    "dataset": "cifar100",
    "fname": "cifar100_l2_AT",
}

cifar100_l2_TRADES_config = {
    **base.l2_TRADES_config,
    "dataset": "cifar100",
    "fname": "cifar100_l2_TRADES",
}

cifar100_l2_MART_config = {
    **base.l2_MART_config,
    "dataset": "cifar100",
    "fname": "cifar100_l2_MART",
}
