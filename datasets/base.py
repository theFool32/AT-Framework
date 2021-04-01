#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod


class Dataset(metaclass=ABCMeta):
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
