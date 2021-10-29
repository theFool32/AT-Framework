#!/usr/bin/env python3

from .at import AT
from .trades import TRADES
from .mart import MART
from .awp import AWP
from .swa import SWA
from .no_defense import NoDefense
from .te import TE
from .rslad import RSLAD
from utils import Configurator

defense_map = {
    "at": AT,
    "trades": TRADES,
    "mart": MART,
    "awp": AWP,
    "swa": SWA,
    "none": NoDefense,
    "te": TE,
    "rslad": RSLAD,
}


def get_defense(model, attack, defense_name=None):
    config = Configurator()
    defense_method = (
        config.defense.lower() if defense_name is None else defense_name.lower()
    )

    if "_" in defense_method:
        inner_defense_name, outer_defense_name = defense_method.split("_")
        inner_defense = get_defense(model, attack, defense_name=inner_defense_name)
        return defense_map[outer_defense_name](model, attack, inner_defense)

    assert defense_method in defense_map
    return defense_map[defense_method](model, attack)


def get_defense_configuration(defense_method: str):
    defense_method = defense_method.lower()
    if "_" in defense_method:
        inner_defense_name, outer_defense_name = defense_method.split("_")
        config = get_defense_configuration(inner_defense_name)
        config.update(defense_map[outer_defense_name].configuration)
        return config

    assert defense_method in defense_map
    return defense_map[defense_method].configuration
