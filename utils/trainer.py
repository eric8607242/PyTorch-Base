import time
import copy
import json
import logging
import random 
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.util import AverageMeter, save, accuracy, min_max_normalize, bn_calibration
from utils.countmacs import MAC_Counter


class Trainer:
    def __init__(self, criterion, optimizer, scheduler, writer, device, CONFIG):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()

        self.writer = writer
        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.CONFIG = CONFIG

        self.epochs = self.CONFIG.epochs

    def train_loop(self, train_loader, test_loader, model):
        best_top1 = 0.0
        for epoch in range(self.epochs):
            logging.info("Learning Rate: {:.4f}".format(self.optimizer.param_groups[0]["lr"]))
            self.writer.add_scalar("learning_rate/weights", self.optimizer.param_groups[0]["lr"], epoch)
            logging.info("Start to train for epoch {}".format(epoch))

            self._training_step(model, train_loader, epoch, info_for_logger="_train_step_", scratch=True)
            self.scheduler.step()

            top1_avg = self.validate(model, test_loader, epoch, scratch=True)
            if best_top1 < top1_avg:
                logging.info("Best top1 acc by now. Save model")
                best_top1 = top1_avg
                save(model, self.optimizer, self.CONFIG.path_to_save_scratch)

        logging.info("The Best top1 acc : {}".format(best_top1))

    def _training_step(self, model, loader, epoch, info_for_logger=""):
        model.train()
        start_time = time.time()
        self.optimizer.zero_grad()

        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()

            outs = model(X)
            loss = self.criterion(outs, y)
            loss.backward()

            self.optimizer.step()
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train="Train")
        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()

    def validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]

                outs = model(X)

                loss = self.criterion(outs, y)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train="val")
        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()

        return top1_avg

    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=""):
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_loss"+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_top1"+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_top5"+info_for_logger, self.top5.get_avg(), epoch)

        top1_avg = self.top1.get_avg()
        logging.info(info_for_logger+val_or_train+":[{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(epoch+1, self.epochs, top1_avg, time.time()-start_time))

    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train, hc_losses=None):
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if (step > 1 and step % self.CONFIG.print_freq==0) or step == len_loader -1 :
            logging.info(val_or_train+
                    ":[{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f}"
                    "Prec@(1, 3) ({:.1%}, {:.1%})".format(
                        epoch+1, self.epochs, step, len_loader-1, self.losses.get_avg(),
                        self.top1.get_avg(), self.top5.get_avg()
                        ))

