#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod


class Defense(metaclass=ABCMeta):
    def __init__(self, _model, _attack, _args):
        self.model = _model
        self.attack = _attack
        self.args = _args

    @abstractmethod
    def train(self, data, label):
        pass

    def test(self, data, label, test_attack=None):
        self.model.eval()
        attack = self.attack
        if test_attack is not None:
            self.attack = test_attack
        result = self.train(data, label)
        self.attack = attack
        return result
