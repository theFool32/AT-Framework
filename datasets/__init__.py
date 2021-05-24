#!/usr/bin/env python3

from .cifar10 import Cifar10
from .cifar100 import Cifar100
from .svhn import SVHN
from .ddpm import DDPM


def get_dataset(args):
    dataset_name = args.dataset.lower()
    if dataset_name == "cifar10":
        return Cifar10(args.data_dir, args.batch_size)
    elif dataset_name == "cifar10_ddpm":
        data_dir = "/".join(args.data_dir.split('/')[:-1]) + 'cifar10'
        cifar10 = Cifar10(data_dir, args.batch_size)
        return DDPM(cifar10, data_dir, args.batch_size)
    elif dataset_name == "cifar100":
        return Cifar100(args.data_dir, args.batch_size)
    elif dataset_name == "svhn":
        return SVHN(args.data_dir, args.batch_size)
    raise NotImplementedError(f"No such dataset: {args.dataset}")
