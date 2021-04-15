#!/usr/bin/env python3

import argparse

import torch
from torch.nn import functional as F
from autoattack import AutoAttack

from models import get_network
from datasets import get_dataset
from attacks import PGD
from utils import AverageMeter

__all__ = ["eval"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PreActResNet18")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data-dir", default="~/datasets/", type=str)
    parser.add_argument(
        "--attack", default="pgd", type=str, choices=["pgd", "fgsm", "free", "none"]
    )
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--attack-iters", default=20, type=int)
    parser.add_argument("--pgd-alpha", default=2, type=float)
    parser.add_argument("--norm", default="l_inf", type=str, choices=["l_inf", "l_2"])
    parser.add_argument("--checkpoint", default="cifar_model_checkpoints", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-amp", default=True, type=bool)
    args = parser.parse_args()
    args.data_dir = f"{args.data_dir}/{args.dataset}"
    return args


def pgd(model, loader, attack, need_nature_acc=False):
    model.eval()
    nat_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.cuda(), label.cuda()

        if need_nature_acc:
            output = model(data)
            pred = torch.max(output, dim=1)[1]
            nat_acc = (pred == label).sum().item()

        adv_data = attack.perturb(data, label).detach()
        adv_output = model(adv_data)
        adv_pred = torch.max(adv_output, dim=1)[1]
        adv_acc = (adv_pred == label).sum().item()

        if need_nature_acc:
            nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
        adv_acc_meter.update(adv_acc / data.size(0), data.size(0))
    return adv_acc_meter.avg, (nat_acc_meter.avg if need_nature_acc else None)


def eval(model, args, dataset, logger=None):
    def out(s):
        if logger is not None:
            logger.info(s)
        else:
            print(s)

    model.eval()

    args.attack_iters = 10
    attack = PGD(args, model=model)
    out("Test pgd10")
    pgd10_acc, nat_acc = pgd(model, dataset.test_loader, attack, need_nature_acc=True)
    out(f"Nature: {nat_acc}")
    out(f"PGD-10: {pgd10_acc}")

    args.attack_iters = 20
    out("Test pgd20")
    attack = PGD(args, model=model)
    pgd20_acc, _ = pgd(model, dataset.test_loader, attack)
    out(f"PGD-20: {pgd20_acc}")

    args.attack_iters = 1
    args.pgd_alpha = 10
    out("Test fgsm")
    attack = PGD(args, model=model)
    fgsm_acc, _ = pgd(model, dataset.test_loader, attack)

    out("=" * 70)
    out(f"Nature: {nat_acc}")
    out(f"FGSM: {fgsm_acc}")
    out(f"PGD-10: {pgd10_acc}")
    out(f"PGD-20: {pgd20_acc}")

    ### Evaluate AutoAttack ###
    l = [x for (x, y) in dataset.test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in dataset.test_loader]
    y_test = torch.cat(l, 0)
    epsilon = 8 / 255.0
    adversary = AutoAttack(model, norm="Linf", eps=epsilon, version="standard")
    adversary.run_standard_evaluation(x_test, y_test, bs=256)


def main():
    args = get_args()
    args.mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1).cuda()
    args.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3, 1, 1).cuda()
    model = get_network(args)
    model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    dataset = get_dataset(args)
    eval(model, args, dataset)


if __name__ == "__main__":
    main()
