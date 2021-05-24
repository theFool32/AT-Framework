#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod
import torch


class Dataset(metaclass=ABCMeta):
    dataset_name = ""

    @property
    @abstractmethod
    def train_loader(self):
        pass

    @property
    @abstractmethod
    def test_loader(self):
        pass

    @property
    def val_loader(self):
        return None

    @property
    def mean(self):
        return torch.tensor((0, 0, 0)).view(3, 1, 1).cuda()

    @property
    def std(self):
        return torch.tensor((1, 1, 1)).view(3, 1, 1).cuda()
