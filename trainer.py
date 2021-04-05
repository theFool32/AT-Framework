#!/usr/bin/env python3


import shutil
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
        defense,
        attack=None,
        scheduler=None,
        writer=None,
    ):
        self.defense = defense
        self.args = args
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.attack = attack
        self.opt = optimizer
        self.scheduler = scheduler
        self.writer = writer

        self.best_acc = -1
        self.best_epoch = -1

    def save_model(self, epoch, acc):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "epoch": epoch,
            },
            os.path.join(self.args.checkpoints, f"model_{epoch}.pth"),
        )
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            target = os.path.abspath(
                os.path.join(self.args.checkpoints, f"model_{epoch}.pth")
            )
            link_name = os.path.abspath(
                os.path.join(self.args.checkpoints, f"model_best.pth")
            )
            try:
                os.symlink(target, link_name)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(link_name)
                    os.symlink(target, link_name)
                else:
                    raise e

    def train_one_epoch(self, epoch):
        self.logger.info("Train_Epoch \tNat_Loss \tNat_Acc \tAdv_Loss \tAdv_Acc")
        self.model.train()
        nat_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        nat_acc_meter = AverageMeter()
        adv_acc_meter = AverageMeter()
        for batch_idx, (data, label) in enumerate(self.dataset.train_loader):
            data, label = data.cuda(), label.cuda()

            output, adv_output, loss, adv_loss, total_loss = self.defense.train(
                data, label
            )

            pred = torch.max(output, dim=1)[1]
            nat_acc = (pred == label).sum().item()

            if adv_output is None:
                adv_acc = 0
            else:
                adv_pred = torch.max(adv_output, dim=1)[1]
                adv_acc = (adv_pred == label).sum().item()

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

            nat_loss_meter.update(loss)
            nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
            adv_loss_meter.update(adv_loss)
            adv_acc_meter.update(adv_acc / data.size(0), data.size(0))

            if batch_idx % self.args.log_step == 0:
                msg = (
                    f"{batch_idx}/{epoch} \t{loss:.3f}/{nat_loss_meter.avg:.3f} \t{nat_acc/data.size(0)*100:.2f}/{nat_acc_meter.avg*100:.2f} "
                    f"\t{adv_loss:.3f}/{adv_loss_meter.avg:.3f} \t{adv_acc/data.size(0)*100:.2f}/{adv_acc_meter.avg*100:.2f}"
                )
                self.logger.info(msg)
        msg = f"{epoch} \t{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f} \t{adv_loss_meter.avg:.3f} \t{adv_acc_meter.avg*100:.2f}"
        self.logger.info(msg)
        self.writer.add_scalar("train/nat_loss", nat_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("train/nat_acc", nat_acc_meter.avg, global_step=epoch)
        self.writer.add_scalar("train/adv_loss", adv_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("train/adv_acc", adv_acc_meter.avg, global_step=epoch)

    def train(self):
        self.model.train()
        for epoch in range(self.args.epoch, self.args.max_epoch):
            if self.scheduler is not None:
                self.scheduler.step()
            self.train_one_epoch(epoch)
            self.test(epoch)

    def test(self, epoch=-1):
        self.model.eval()
        self.logger.info("=" * 70)
        self.logger.info("Test_Epoch \tNat_Loss \tNat_Acc \tAdv_Loss \tAdv_Acc")
        nat_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        nat_acc_meter = AverageMeter()
        adv_acc_meter = AverageMeter()
        for batch_idx, (data, label) in enumerate(self.dataset.test_loader):
            data, label = data.cuda(), label.cuda()

            output, adv_output, loss, adv_loss, total_loss = self.defense.test(
                data, label
            )

            pred = torch.max(output, dim=1)[1]
            nat_acc = (pred == label).sum().item()

            adv_pred = torch.max(adv_output, dim=1)[1]
            adv_acc = (adv_pred == label).sum().item()

            nat_loss_meter.update(loss)
            nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
            adv_loss_meter.update(adv_loss)
            adv_acc_meter.update(adv_acc / data.size(0), data.size(0))

        msg = f"{epoch} \t{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f} \t{adv_loss_meter.avg:.3f} \t{adv_acc_meter.avg*100:.2f}"
        self.logger.info("=" * 70)
        self.writer.add_scalar("test/nat_loss", nat_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("test/nat_acc", nat_acc_meter.avg, global_step=epoch)
        self.writer.add_scalar("test/adv_loss", adv_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar("test/adv_acc", adv_acc_meter.avg, global_step=epoch)
        self.writer.flush()
        self.logger.info(msg)
        self.save_model(epoch, adv_acc_meter.avg)

    def val(self):
        self.model.eval()
        pass
