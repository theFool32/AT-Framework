#!/usr/bin/env python3


import torch
from torch.nn import functional as F
import logging

from datasets.base import Dataset
from utils import AverageMeter


class Trainer:
    def __init__(
        self,
        args,
        model,
        dataset: Dataset,
        logger: logging.Logger,
        optimizer,
        attack=None,
        scheduler=None,
        writer=None,
    ):
        self.args      = args
        self.model     = model
        self.dataset   = dataset
        self.logger    = logger
        self.attack    = attack
        self.opt       = optimizer
        self.scheduler = scheduler

    def train_one_epoch(self, epoch):
        self.logger.info("Epoch \tNat_Loss \tNat_Acc")
        nat_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        nat_acc_meter  = AverageMeter()
        adv_acc_meter  = AverageMeter()
        for batch_idx, (data, label) in enumerate(self.dataset.train_loader):
            # TODO: device
            data, label = data.cuda(), label.cuda()

            output = self.model(data)
            pred = torch.max(output, dim=1)[1]
            nat_acc = (pred == label).sum().item()

            loss = F.cross_entropy(output, label)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            nat_loss_meter.update(loss.item())
            nat_acc_meter.update(nat_acc/data.size(0), data.size(0))

            if batch_idx % self.args.log_step == 0:
                msg = f"{batch_idx}/{epoch} \t{loss.item():.3f}/{nat_loss_meter.avg:.3f} \t{nat_acc/data.size(0)*100:.2f}/{nat_acc_meter.avg*100:.2f}"
                self.logger.info(msg)
        msg = f"{epoch} \t{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f}"
        self.logger.info(msg)

    def train(self):
        for epoch in range(self.args.epoch, self.args.max_epoch):
            if self.scheduler is not None:
                self.scheduler.step()
            self.train_one_epoch(epoch)

    def test(self):
        pass

    def val(self):
        pass
