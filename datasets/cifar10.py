#!/usr/bin/env python3

from torchvision import datasets
from torch.utils import data
from torchvision import transforms

from .base import Dataset


class Cifar10(Dataset):
    def __init__(self, root, batch_size=128, val=False):
        """
        load cifar10 dataset using torchvision
        ---
        val: Bool
            use `https://github.com/locuslab/robust_overfitting/blob/master/generate_validation.py` for validation if true
        """
        self.val = val

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        if val:
            # TODO: add code for `generate_validation.py`
            raise NotImplementedError("Validation for cifar10 not implemented.")
        else:
            self._train_dataset = datasets.CIFAR10(
                root=root, train=True, download=True, transform=train_transform
            )
            self._test_dataset = datasets.CIFAR10(
                root=root, train=False, download=True, transform=test_transform
            )

        self._train_loader = data.DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
        )
        self._test_loader = data.DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            num_workers=2,
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
        if self.val:
            return self._val_loader
        return None


if __name__ == "__main__":
    cifar10 = Cifar10("~/datasets/cifar10")
    assert len(cifar10.train_loader.dataset) == 50000
    assert len(cifar10.test_loader.dataset) == 10000
