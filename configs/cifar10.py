#!/usr/bin/env python3

from . import base

cifar10_linf_AT_config = {
    **base.linf_AT_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_AT",
}

cifar10_linf_TRADES_config = {
    **base.linf_TRADES_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_TRADES",
}

cifar10_linf_MART_config = {
    **base.linf_MART_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_MART",
}

cifar10_l2_AT_config = {
    **base.l2_AT_config,
    "dataset": "cifar10",
    "fname": "cifar10_l2_AT",
}

cifar10_l2_TRADES_config = {
    **base.l2_TRADES_config,
    "dataset": "cifar10",
    "fname": "cifar10_l2_TRADES",
}

cifar10_l2_MART_config = {
    **base.l2_MART_config,
    "dataset": "cifar10",
    "fname": "cifar10_l2_MART",
}

# ====================================
# AWP
cifar10_linf_AWP_AT_config = {
    **base.linf_AT_config,
    **base.linf_AWP_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_AWP_AT",
}

cifar10_linf_AWP_TRADES_config = {
    **base.linf_TRADES_config,
    **base.linf_AWP_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_AWP_TRADES",
}

cifar10_linf_AWP_MART_config = {
    **base.linf_MART_config,
    **base.linf_AWP_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_AWP_MART",
}

# ====================================
# SWA
cifar10_linf_SWA_AT_config = {
    **base.linf_AT_config,
    **base.linf_SWA_config,
    "dataset": "cifar10",
    "fname": "cifar10_linf_SWA_AT",
}
