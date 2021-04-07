#!/usr/bin/env python3

from .at import AT
from .trades import TRADES
from .no_defense import NoDefense

def get_defense(args, model, attack):
    defense_method = args.defense.lower()
    if defense_method == 'at':
        return AT(model, attack, **args.__dict__)
    elif defense_method == 'trades':
        return TRADES(model, attack, **args.__dict__)
    else:
        raise NotImplementedError(f"Defense not implemented: {defense_method}")
