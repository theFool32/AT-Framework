#!/usr/bin/env python3

from torch.utils import data
from . import base

svhn_linf_config = {
    "pgd_alpha": 1,
    "dataset": "svhn",
}

svhn_linf_AT_config = {
    **base.linf_AT_config,
    **svhn_linf_config,
    "fname": "svhn_linf_AT",
}

svhn_linf_TRADES_config = {
    **base.linf_TRADES_config,
    **svhn_linf_config,
    "fname": "svhn_linf_TRADES",
}

svhn_linf_MART_config = {
    **base.linf_MART_config,
    **svhn_linf_config,
    "fname": "svhn_linf_MART",
}

svhn_l2_AT_config = {
    **base.l2_AT_config,
    "dataset": "svhn",
    "fname": "svhn_l2_AT",
}

svhn_l2_TRADES_config = {
    **base.l2_TRADES_config,
    "dataset": "svhn",
    "fname": "svhn_l2_TRADES",
}

svhn_l2_MART_config = {
    **base.l2_MART_config,
    "dataset": "svhn",
    "fname": "svhn_l2_MART",
}
