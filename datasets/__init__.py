#!/usr/bin/env python3

from .cifar10 import Cifar10

def get_dataset(args):
    if args.dataset == 'cifar10':
        return Cifar10(args.data_dir, args.batch_size) # TODO: val
    raise NotImplementedError(f"No such dataset: {args.dataset}")
