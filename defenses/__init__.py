#!/usr/bin/env python3

from .at import AT
from .no_defense import NoDefense

def get_defense(args, model, attack):
    return AT(model, attack)
