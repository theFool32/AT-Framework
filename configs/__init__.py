#!/usr/bin/env python3

from datasets import get_dataset_configuration
from defenses import get_defense_configuration

__all__ = ["get_configuration"]

base_config = {
    "log_step": 100,
    "batch_size": 128,
    "save_checkpoints": lambda epoch: True,
    "weight_decay": 5e-4,
}

def get_configuration(config_name: str):
    dataset, lp, defense = config_name.split(":")

    dataset_config = get_dataset_configuration(dataset)[lp]
    defense_config = get_defense_configuration(defense)

    base_config.update(dataset_config)
    base_config.update(defense_config)
    base_config.update({"norm": lp})

    return base_config
