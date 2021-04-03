#!/usr/bin/env python3

import argparse

import time
import logging
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

from trainer import Trainer
from models import get_network
from datasets import get_dataset
from attacks import *
from utils import git_version
from defenses import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PreActResNet18")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data-dir", default="~/datasets/cifar10", type=str)
    parser.add_argument("--max-epoch", default=100, type=int)
    parser.add_argument("--epoch", default=0, type=int)
    parser.add_argument(
        "--attack", default="pgd", type=str, choices=["pgd", "fgsm", "free", "none"]
    )
    parser.add_argument("--log-step", default=100, type=int)
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--attack-iters", default=10, type=int)
    parser.add_argument("--pgd-alpha", default=2, type=float)
    parser.add_argument("--norm", default="l_inf", type=str, choices=["l_inf", "l_2"])
    parser.add_argument("--fname", default="cifar_model", type=str)
    parser.add_argument("--checkpoints", default="cifar_model_checkpoints", type=str)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--resume", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--chkpt-iters", default=10, type=int)
    args = parser.parse_args()
    # args = vars(args)
    return args


def main():
    args = get_args()

    current_time = time.ctime()
    args.fname = args.fname + '/' + current_time
    args.checkpoints = args.checkpoints + '/' + current_time

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(
                os.path.join(args.fname, "eval.log" if args.eval else "output.log")
            ),
            logging.StreamHandler(),
        ],
    )
    logger.info(args)
    logger.info(git_version())
    writer = SummaryWriter(args.fname)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = get_dataset(args)

    args.mean = dataset.mean
    args.std = dataset.std
    model = get_network(args)


    # TODO:
    opt = torch.optim.SGD(
        model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[75, 90], gamma=0.1
    )

    attack = PGD(args, model=model)
    defense = get_defense(args, model, attack)

    trainer = Trainer(
        args=args,
        model=model,
        dataset=dataset,
        logger=logger,
        optimizer=opt,
        scheduler=scheduler,
        attack=attack,
        writer=writer,
        defense=defense
    )
    trainer.train()


if __name__ == "__main__":
    main()
