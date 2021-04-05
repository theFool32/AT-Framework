#!/usr/bin/env python3


base_config = {
    'log_step': 100,
    'max_epoch': 110,
    'batch_size': 128,
}

linf_base_config = {
    **base_config,
    'lr': 1e-1,
    'weight-decay': 5e-4,
    'epsilon': 8,
    'norm': 'l_inf',
}


linf_AT_config = {
    **linf_base_config,
    'attack': 'pgd',
    'inner_loss': 'CE',
    'outer_loss': 'CE',
    'attack_iters': 10,
    'pgd_alpha': 2,
}
