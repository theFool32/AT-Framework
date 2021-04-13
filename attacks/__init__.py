#!/usr/bin/env python3

from .pgd import PGD, PGD_Test


__all__ = ["get_attack"]

def get_attack(args):
    attack_name = args.attack.lower()
    if attack_name == "pgd":
        return PGD
    else:
        raise NotImplementedError(f"Attack not implemented: {attack_name}")
