#!/usr/bin/env python3

import torch
from torchvision import datasets
from torch.utils import data
from torchvision import transforms

from .base import Dataset


class Cifar100(Dataset):
    num_classes = 100
    dataset_name = "cifar100"
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
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self._train_dataset = datasets.CIFAR100(
            root=root, train=True, download=True, transform=train_transform
        )
        self._test_dataset = datasets.CIFAR100(
            root=root, train=False, download=True, transform=test_transform
        )

        self._train_loader = data.DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
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
        # https://github.com/csdongxian/AWP/blob/main/AT_AWP/train_cifar100.py
        return (
            torch.tensor((0.5070751592371323, 0.48654887331495095, 0.4409178433670343))
            .view(3, 1, 1)
            .cuda()
        )

    @property
    def std(self):
        return (
            torch.tensor((0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            .view(3, 1, 1)
            .cuda()
        )
