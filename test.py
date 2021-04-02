#!/usr/bin/env python3

import argparse

import torch
from torch.nn import functional as F

from models import get_network
from datasets import get_dataset
from attacks import PGD
from utils import AverageMeter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PreActResNet18")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data-dir", default="~/datasets/cifar10", type=str)
    parser.add_argument(
        "--attack", default="pgd", type=str, choices=["pgd", "fgsm", "free", "none"]
    )
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--attack-iters", default=20, type=int)
    parser.add_argument("--pgd-alpha", default=2, type=float)
    parser.add_argument("--norm", default="l_inf", type=str, choices=["l_inf", "l_2"])
    parser.add_argument("--checkpoint", default="cifar_model_checkpoints", type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1).cuda()
    args.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3, 1, 1).cuda()
    model = get_network(args)
    model.load_state_dict(torch.load(args.checkpoint)['state'])
    attack = PGD(args, model=model)
    dataset = get_dataset(args)
    model.eval()
    print("Test_Epoch \tNat_Loss \tNat_Acc \tAdv_Loss \tAdv_Acc")
    nat_loss_meter = AverageMeter()
    adv_loss_meter = AverageMeter()
    nat_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()
    for batch_idx, (data, label) in enumerate(dataset.test_loader):
        data, label = data.cuda(), label.cuda()

        output = model(data)
        pred = torch.max(output, dim=1)[1]
        nat_acc = (pred == label).sum().item()

        adv_data = attack.perturb(data).detach()
        adv_output = model(adv_data)
        adv_pred = torch.max(adv_output, dim=1)[1]
        adv_acc = (adv_pred == label).sum().item()

        loss = F.cross_entropy(output, label)
        adv_loss = F.cross_entropy(adv_output, label)

        nat_loss_meter.update(loss.item())
        nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
        adv_loss_meter.update(adv_loss.item())
        adv_acc_meter.update(adv_acc / data.size(0), data.size(0))

    msg = f"{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f} \t{adv_loss_meter.avg:.3f} \t{adv_acc_meter.avg*100:.2f}"
    print(msg)


if __name__ == '__main__':
    main()
