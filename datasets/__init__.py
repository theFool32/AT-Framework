#!/usr/bin/env python3

from .cifar10 import Cifar10
from .cifar100 import Cifar100
from .svhn import SVHN
from .ddpm import DDPM

dataset_map = {
    "cifar10": Cifar10,
    "cifar100": Cifar100,
    "svhn": SVHN,
    "ddpm": DDPM,
}


def get_dataset(dataset_name):
    from utils import Configurator
    dataset_name = dataset_name.lower()
    config = Configurator()

    # FIXME: currently only support cifar10_ddpm
    if "_" in dataset_name:
        data_dir = "/".join(config.data_dir.split("/")[:-1]) + "cifar10"
        cifar10 = Cifar10(data_dir, config.batch_size)
        return DDPM(cifar10, data_dir, config.batch_size)

    assert dataset_name in dataset_map
    return dataset_map[dataset_name](config.data_dir, config.batch_size)

def get_dataset_configuration(dataset_name):
    dataset_name = dataset_name.lower()

    # FIXME: currently only support cifar10_ddpm
    if "_" in dataset_name:
        data_dir = "/".join(config.data_dir.split("/")[:-1]) + "cifar10"
        config = Cifar10.configuration
        config.update(DDPM.configuration)
        return config

    assert dataset_name in dataset_map
    return dataset_map[dataset_name].configuration
