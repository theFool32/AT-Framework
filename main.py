#!/usr/bin/env python3

import random
import argparse
import time
import logging
import os
import numpy as np
import torch

from trainer import Trainer
from models import get_network
from datasets import get_dataset
from attacks import get_attack
from utils import git_version
from utils import Configurator
from utils import Lr_schedule
from utils import get_args
from defenses import get_defense

def main():
    args = get_args()

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(args.fname[8:])
    else:
        import wandb

        wandb.init(
            project=args.project,
            name=args.fname.replace("/", "_")[8:],
            config=args.__dict__,
            settings=wandb.Settings(_disable_stats=True),
        )
        writer = None

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, "output.log")),
            logging.StreamHandler(),
        ],
    )

    logger.info(args)
    logger.info(git_version())

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = get_dataset(args.dataset)

    args.mean = dataset.mean
    args.std = dataset.std
    args.dataset = dataset
    model = get_network(args.model)

    opt = torch.optim.SGD(
        model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = Lr_schedule(
        opt, milestones=[int(v) for v in args.lr_adjust.split(",")], gamma=0.1
    )

    if args.resume_checkpoint != "":
        state = torch.load(args.resume_checkpoint)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        args.epoch = state["epoch"] + 1
    else:
        args.epoch = 1

    if not args.no_amp:
        # from torch.cuda.amp.grad_scaler import GradScaler
        # scaler = GradScaler()
        # args.scaler = scaler

        from apex import amp

        model, opt = amp.initialize(
            model, opt, opt_level="O1", loss_scale=1.0, verbosity=False
        )
        args.amp = amp

    args.opt = opt

    attack = get_attack(model=model)
    defense = get_defense(model, attack)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        logger=logger,
        optimizer=opt,
        scheduler=scheduler,
        attack=attack,
        writer=writer,
        defense=defense,
    )
    trainer.train()

    # logger.info("Begin evaluating last")
    # eval(model, args, dataset, logger)

    if not args.tensorboard:
        wandb.finish()


if __name__ == "__main__":
    main()
