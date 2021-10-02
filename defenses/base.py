#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod


class Defense(metaclass=ABCMeta):
    configuration = {}

    def __init__(self, _model, _attack):
        self.model = _model
        self.attack = _attack

    @abstractmethod
    def train(self, data, label):
        pass

    def postprocess(self, epoch):
        pass

    def test(self, data, label, test_attack=None):
        self.model.eval()
        attack = self.attack
        if test_attack is not None:
            self.attack = test_attack
        result = self.train(data, label)
        self.attack = attack
        return result
