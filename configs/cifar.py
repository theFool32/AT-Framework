#!/usr/bin/env python3

from . import base

__all__ = ['cifar10_linf_AT_config']

cifar10_linf_AT_config = {
    **base.linf_AT_config,
    'dataset': 'cifar10',
}
