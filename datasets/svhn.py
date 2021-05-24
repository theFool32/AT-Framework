#!/usr/bin/env python3

import torch
from torchvision import datasets
from torch.utils import data
from torchvision import transforms

from .base import Dataset


class SVHN(Dataset):
    num_classes = 10
    dataset_name = "svhn"

    def __init__(self, root, batch_size=128):
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self._train_dataset = datasets.SVHN(
            root=root, split="train", download=True, transform=train_transform
        )
        self._test_dataset = datasets.SVHN(
            root=root, split="test", download=True, transform=test_transform
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
        return torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1).cuda()

    @property
    def std(self):
        return torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1).cuda()
