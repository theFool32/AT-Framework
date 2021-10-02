#!/usr/bin/env python3

from .pgd import PGD, PGD_Test
from utils import Configurator

__all__ = ["get_attack"]


def get_attack(model):
    attack_name = Configurator().attack.lower()
    if attack_name == "pgd":
        return PGD(model)
    else:
        raise NotImplementedError(f"Attack not implemented: {attack_name}")
