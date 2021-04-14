#!/usr/bin/env python3

from .at import AT
from .trades import TRADES
from .mart import MART
from .awp import AWP
from .no_defense import NoDefense


def get_defense(args, model, attack, defense_name=None):
    defense_method = (
        args.defense.lower() if defense_name is None else defense_name.lower()
    )
    if defense_method == "at":
        return AT(model, attack, args)
    elif defense_method == "trades":
        return TRADES(model, attack, args)
    elif defense_method == "mart":
        return MART(model, attack, args)
    elif defense_method == "none":
        return NoDefense(model, attack, args)
    elif defense_method.startswith("awp"):
        inner_defense_name = defense_method[4:]
        inner_defense = get_defense(
            args, model, attack, defense_name=inner_defense_name
        )
        return AWP(model, attack, args, inner_defense)
    else:
        raise NotImplementedError(f"Defense not implemented: {defense_method}")
