#!/usr/bin/env python3


import errno
from os.path import commonprefix
import shutil
import os
import torch
from torch.nn import functional as F
import logging

import wandb

from datasets.base import Dataset
from utils import AverageMeter
from utils import Configurator
from attacks import PGD_Test


class Trainer:
    def __init__(
        self,
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
        self.config = Configurator()
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.attack = attack
        self.opt = optimizer
        self.scheduler = scheduler
        self.writer = writer

        self.best_acc = -1
        self.best_epoch = -1

        self.test_attack = PGD_Test(model, iters=10)

    def save_model(self, epoch, adv_acc=None, nat_acc=None, only_best_and_last=True):
        if only_best_and_last:
            if adv_acc is not None and adv_acc > self.best_acc:
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.config.checkpoints, f"model_best.pth"),
                )
            if epoch == self.config.max_epoch:
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.config.checkpoints, f"model_last.pth"),
                )
        else:
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(self.config.checkpoints, f"model_{epoch}.pth"),
            )
            if adv_acc is not None and adv_acc > self.best_acc:
                target = os.path.abspath(
                    os.path.join(self.config.checkpoints, f"model_{epoch}.pth")
                )
                link_name = os.path.abspath(
                    os.path.join(self.config.checkpoints, "model_best.pth")
                )
                try:
                    os.symlink(target, link_name)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        os.remove(link_name)
                        os.symlink(target, link_name)
                    else:
                        raise e

        if adv_acc is not None and adv_acc > self.best_acc:
            self.best_acc = adv_acc
            self.best_epoch = epoch
            if not self.config.tensorboard:
                wandb.run.summary["best_adv_acc"] = adv_acc
                if nat_acc is not None:
                    wandb.run.summary["best_nat_acc"] = nat_acc

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

            if isinstance(total_loss, torch.Tensor):
                self.opt.zero_grad()
                if not self.config.no_amp:
                    # self.config.scaler.scale(total_loss).backward()
                    # self.config.scaler.step(self.opt)
                    # self.config.scaler.update()
                    with self.config.amp.scale_loss(
                        total_loss, self.config.opt
                    ) as scaled_loss:
                        scaled_loss.backward()
                    self.opt.step()
                else:
                    total_loss.backward()
                    self.opt.step()

            nat_loss_meter.update(loss)
            nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
            adv_loss_meter.update(adv_loss)
            adv_acc_meter.update(adv_acc / data.size(0), data.size(0))

            if batch_idx % self.config.log_step == 0:
                msg = (
                    f"{batch_idx}/{epoch} \t{loss:.3f}/{nat_loss_meter.avg:.3f} \t{nat_acc/data.size(0)*100:.2f}/{nat_acc_meter.avg*100:.2f} "
                    f"\t{adv_loss:.3f}/{adv_loss_meter.avg:.3f} \t{adv_acc/data.size(0)*100:.2f}/{adv_acc_meter.avg*100:.2f}"
                )
                self.logger.info(msg)
        msg = f"{epoch} \t{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f} \t{adv_loss_meter.avg:.3f} \t{adv_acc_meter.avg*100:.2f}"
        self.logger.info(msg)
        if self.config.tensorboard:
            self.writer.add_scalar(
                "train/nat_loss", nat_loss_meter.avg, global_step=epoch
            )
            self.writer.add_scalar(
                "train/nat_acc", nat_acc_meter.avg, global_step=epoch
            )
            self.writer.add_scalar(
                "train/adv_loss", adv_loss_meter.avg, global_step=epoch
            )
            self.writer.add_scalar(
                "train/adv_acc", adv_acc_meter.avg, global_step=epoch
            )
        else:
            wandb.log(
                {
                    "train/nat_loss": nat_loss_meter.avg,
                    "train/nat_acc": nat_acc_meter.avg,
                    "train/adv_loss": adv_loss_meter.avg,
                    "train/adv_acc": adv_acc_meter.avg,
                    "epoch": epoch,
                }
            )

    def train(self):
        self.model.train()
        for epoch in range(self.config.epoch, self.config.max_epoch + 1):
            self.model.test_mode(False, self.config.no_amp)
            self.config.epoch = epoch
            self.train_one_epoch(epoch)
            self.defense.postprocess(epoch)
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            if (
                self.config.save_checkpoints is not None
                and self.config.save_checkpoints(epoch)
            ):
                self.model.test_mode(True)
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
                data, label, self.test_attack
            )

            pred = torch.max(output, dim=1)[1]
            nat_acc = (pred == label).sum().item()

            adv_pred = torch.max(adv_output, dim=1)[1]
            adv_acc = (adv_pred == label).sum().item()

            nat_loss_meter.update(loss)
            nat_acc_meter.update(nat_acc / data.size(0), data.size(0))
            adv_loss_meter.update(adv_loss)
            adv_acc_meter.update(adv_acc / data.size(0), data.size(0))

        self.save_model(epoch, adv_acc_meter.avg, nat_acc_meter.avg, True)
        msg = f"{epoch} \t{nat_loss_meter.avg:.3f} \t{nat_acc_meter.avg*100:.2f} \t{adv_loss_meter.avg:.3f} \t{adv_acc_meter.avg*100:.2f}"
        self.logger.info(msg)
        self.logger.info(f"Best: {self.best_epoch} \t{self.best_acc*100:.2f}")
        self.logger.info("=" * 70)

        if self.config.tensorboard:
            self.writer.add_scalar(
                "test/nat_loss", nat_loss_meter.avg, global_step=epoch
            )
            self.writer.add_scalar("test/nat_acc", nat_acc_meter.avg, global_step=epoch)
            self.writer.add_scalar(
                "test/adv_loss", adv_loss_meter.avg, global_step=epoch
            )
            self.writer.add_scalar("test/adv_acc", adv_acc_meter.avg, global_step=epoch)
            self.writer.flush()
        else:
            wandb.run.summary["last_nat_acc"] = nat_acc_meter.avg
            wandb.run.summary["last_adv_acc"] = adv_acc_meter.avg
            wandb.log(
                {
                    "test/nat_loss": nat_loss_meter.avg,
                    "test/nat_acc": nat_acc_meter.avg,
                    "test/adv_loss": adv_loss_meter.avg,
                    "test/adv_acc": adv_acc_meter.avg,
                    "epoch": epoch,
                }
            )

    def val(self):
        self.model.eval()
        pass
