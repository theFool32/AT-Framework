#!/usr/bin/env python3

from collections import namedtuple

import torch
from torchvision import datasets
from torch.utils import data
from torchvision import transforms
from torch import Tensor
import numpy as np

from .base import Dataset
from .random_sampler import RandomSampler

__all__ = ["Cifar10"]

class Cifar10(Dataset):
    num_classes = 10
    dataset_name = "cifar10"
    configuration = {
        "l2": {
            "lr": 1e-1,
            "epsilon": 128,
            "pgd_alpha": 15,
        },
        "linf": {
            "lr": 1e-1,
            "epsilon": 8,
            "pgd_alpha": 2,
        }
    }

    def __init__(self, root, batch_size=128):
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


        self._train_dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )
        self._test_dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )

        self.random_sampler = RandomSampler(self._train_dataset, generator=None)
        self._train_loader = data.DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            # shuffle=True,
            sampler=self.random_sampler,
        )
        self._test_loader = data.DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def val_loader(self):
        return None

    @property
    def mean(self):
        return torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1).cuda()

    @property
    def std(self):
        return torch.tensor((0.2471, 0.2435, 0.2616)).view(3, 1, 1).cuda()


if __name__ == "__main__":
    cifar10 = Cifar10("~/datasets/cifar10")
    assert len(cifar10.train_loader.dataset) == 50000
    assert len(cifar10.test_loader.dataset) == 10000
