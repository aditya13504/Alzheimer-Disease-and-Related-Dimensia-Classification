# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np

from .evaluation import decode_preds, compute_nme

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, criterion, optimizer, epoch, writer_dict, device=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    if device is None:
        device = next(model.parameters()).device
    end = time.time()

    for i, (inp, target) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp.to(device))
        target = target.to(device)
        loss = criterion(output, target)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                  f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  f'Loss {losses.val:.5f} ({losses.avg:.5f})'
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict, device=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()
    if device is None:
        device = next(model.parameters()).device  # Get the device the model is on

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp.to(device))
            target = target.to(device)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += nme_temp.sum()
            nme_count += preds.size(0)
            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            if i % config.PRINT_FREQ == 0:
                msg = f'Val: [{i}/{len(val_loader)}]\t' \
                      f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      f'Loss {losses.val:.5f} ({losses.avg:.5f})'
                logger.info(msg)

                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['valid_global_steps']
                    writer.add_scalar('val_loss', losses.val, global_steps)
                    writer_dict['valid_global_steps'] = global_steps + 1
            end = time.time()

    nme = nme_batch_sum / nme_count if nme_count > 0 else 0

    msg = 'Val time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(batch_time.avg, losses.avg, nme)
    logger.info(msg)

    return nme, predictions


def inference(config, data_loader, model, device=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    if device is None:
        device = next(model.parameters()).device  # Get the device the model is on
    outputs_list = []
    targets_list = []
    end = time.time()

    with torch.no_grad():
        for i, (inp, target) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp.to(device))
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())
            batch_time.update(time.time() - end)
            end = time.time()

    outputs = torch.cat(outputs_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return outputs, targets



