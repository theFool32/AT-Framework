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
from utils import Parameters
from defenses import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cifar10_linf_AT", type=str)
    parser.add_argument("--model", default="PreActResNet18")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-dir", default="~/datasets/cifar10", type=str)
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--defense", type=str)
    parser.add_argument("--attack", type=str, choices=["pgd", "fgsm", "free", "none"])
    parser.add_argument("--inner-loss", type=str)
    parser.add_argument("--outer-loss", type=str)
    parser.add_argument("--log-step", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epsilon", type=int)
    parser.add_argument("--attack-iters", type=int)
    parser.add_argument("--pgd-alpha", type=float)
    parser.add_argument("--norm", type=str, choices=["l_inf", "l_2"])
    parser.add_argument("--fname", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume-checkpoint", default="", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import configs

    try:
        config = getattr(configs, args.config + "_config")
        args = vars(args)
        args = {**config, **{k: args[k] for k in args if args[k] is not None}}
        args = Parameters(args)
    except Exception:
        raise NotImplementedError(f"No such configuration: {args.config}")
    args.checkpoints = args.fname + "_checkpoints"

    current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    args.fname = args.fname + "/" + current_time
    args.checkpoints = args.checkpoints + "/" + current_time
    return args


def main():
    args = get_args()

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
        opt, milestones=[100, 105], gamma=0.1
    )

    attack = PGD(args, model=model)
    defense = get_defense(args, model, attack)

    if args.resume_checkpoint != "":
        state = torch.load(args.resume_checkpoint)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        args.epoch = state["epoch"]

    trainer = Trainer(
        args=args,
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


if __name__ == "__main__":
    main()
