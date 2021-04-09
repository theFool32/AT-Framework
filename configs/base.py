#!/usr/bin/env python3


from torchvision.transforms.transforms import Lambda

base_config = {
    "log_step": 100,
    "max_epoch": 100,
    "batch_size": 128,
    "epoch": 0,
    "save_checkpoints": lambda epoch: epoch % 10 == 0 or epoch >= 70,
}

linf_base_config = {
    **base_config,
    "lr": 1e-1,
    "weight_decay": 5e-4,
    "epsilon": 8,
    "norm": "l_inf",
}


linf_AT_config = {
    **linf_base_config,
    "attack": "pgd",
    "defense": "at",
    "inner_loss": "CE",
    "outer_loss": "CE",
    "attack_iters": 10,
    "pgd_alpha": 2,
}

linf_TRADES_config = {
    **linf_AT_config,
    "defense": "trades",
    "attack": "pgd",
    "inner_loss": "trades_inner",
    "outer_loss": "trades_outer",
}

linf_MART_config = {
    **linf_AT_config,
    "defense": "mart",
    "attack": "pgd",
    "inner_loss": "CE",
    "outer_loss": "mart_outer",
}
