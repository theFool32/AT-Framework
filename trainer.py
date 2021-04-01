#!/usr/bin/env python3


import os
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
        self.args = args
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.attack = attack
        self.opt = optimizer
        self.scheduler = scheduler
        self.writer = writer

    def save_model(self, epoch, name=""):
        torch.save(
            {"state": self.model.state_dict(), "optimizer": self.opt.state_dict()},
            os.path.join(self.args.checkpoints, f"model_{epoch}_{name}.pth"),
        )

    def train_one_epoch(self, epoch):
        self.logger.info("Train_Epoch \tNat_Loss \tNat_Acc \tAdv_Loss \tAdv_Acc")
        self.model.train()
        nat_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        nat_acc_meter = AverageMeter()
        adv_acc_meter = AverageMeter()
        for batch_idx, (data, label) in enumerate(self.dataset.train_loader):
            # TODO: device
            data, label = data.cuda(), label.cuda()

            output = self.model(data)
            pred = torch.max(output, dim=1)[1]
            nat_acc = (pred == label).sum().item()

            adv_data = self.attack.perturb(data).detach()
            adv_output = self.model(adv_data)
            adv_pred = torch.max(adv_output, dim=1)[1]
            adv_acc = (adv_pred == label).sum().item()

            loss = F.cross_entropy(output, label)
            adv_loss = F.cross_entropy(adv_output, label)

            self.opt.zero_grad()
            # loss.backward()
            adv_loss.backward()
            self.opt.step()

            nat_loss_meter.update(loss.item())
            nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
            adv_loss_meter.update(adv_loss.item())
            adv_acc_meter.update(adv_acc / data.size(0), data.size(0))

            if batch_idx % self.args.log_step == 0:
                msg = (
                    f"{batch_idx}/{epoch} \t{loss.item():.3f}/{nat_loss_meter.avg:.3f} \t{nat_acc/data.size(0)*100:.2f}/{nat_acc_meter.avg*100:.2f} "
                    f"\t{adv_loss.item():.3f}/{adv_loss_meter.avg:.3f} \t{adv_acc/data.size(0)*100:.2f}/{adv_acc_meter.avg*100:.2f}"
                )
                self.logger.info(msg)
        msg = f"{epoch} \t{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f} \t{adv_loss_meter.avg:.3f} \t{adv_acc_meter.avg*100:.2f}"
        self.logger.info(msg)
        self.writer.add_scalar("train/nat_loss", nat_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("train/nat_acc", nat_acc_meter.avg, global_step=epoch)
        self.writer.add_scalar("train/adv_loss", adv_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("train/adv_acc", adv_acc_meter.avg, global_step=epoch)
        self.test(epoch)

    def train(self):
        self.model.train()
        for epoch in range(self.args.epoch, self.args.max_epoch):
            if self.scheduler is not None:
                self.scheduler.step()
            self.train_one_epoch(epoch)

    def test(self, epoch=-1):
        self.model.eval()
        self.logger.info("Test_Epoch \tNat_Loss \tNat_Acc \tAdv_Loss \tAdv_Acc")
        nat_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        nat_acc_meter = AverageMeter()
        adv_acc_meter = AverageMeter()
        for batch_idx, (data, label) in enumerate(self.dataset.test_loader):
            data, label = data.cuda(), label.cuda()

            output = self.model(data)
            pred = torch.max(output, dim=1)[1]
            nat_acc = (pred == label).sum().item()

            adv_data = self.attack.perturb(data).detach()
            adv_output = self.model(adv_data)
            adv_pred = torch.max(adv_output, dim=1)[1]
            adv_acc = (adv_pred == label).sum().item()

            loss = F.cross_entropy(output, label)
            adv_loss = F.cross_entropy(adv_output, label)

            nat_loss_meter.update(loss.item())
            nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
            adv_loss_meter.update(adv_loss.item())
            adv_acc_meter.update(adv_acc / data.size(0), data.size(0))

        msg = f"{epoch} \t{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f} \t{adv_loss_meter.avg:.3f} \t{adv_acc_meter.avg*100:.2f}"
        self.writer.add_scalar("test/nat_loss", nat_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("test/nat_acc", nat_acc_meter.avg, global_step=epoch)
        self.writer.add_scalar("test/adv_loss", adv_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("test/adv_acc", adv_acc_meter.avg, global_step=epoch)
        self.writer.flush()
        self.logger.info(msg)
        self.save_model(epoch)

    def val(self):
        self.model.eval()
        pass
