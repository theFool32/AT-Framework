#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod


class Defense(metaclass=ABCMeta):
    def __init__(self, model, attack, **kw):
        self.model = model
        self.attack = attack

    @abstractmethod
    def train(self, data, label):
        pass

    def test(self, data, label):
        self.model.eval()
        return self.train(data, label)
