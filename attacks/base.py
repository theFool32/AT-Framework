#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod


class Attack(metaclass=ABCMeta):
    @abstractmethod
    def perturbe(self):
        pass
