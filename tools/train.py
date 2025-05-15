# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
from tools.mod_hrnet import get_hrnet_ad_model


class BinaryTargetTransform:
    def __init__(self, class_list):
        self.class_list = class_list
        self.ad_classes = ["Very Mild Demented", "Mild Demented", "Moderate Demented"]
    def __call__(self, idx):
        return int(self.class_list[idx] in self.ad_classes)


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    tsfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_ds = datasets.ImageFolder(
        root="./train",
        transform=tsfm
    )
    train_ds.target_transform = BinaryTargetTransform(train_ds.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_hrnet_ad_model(pretrained=True, single_channel=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = utils.get_optimizer(config, model)
    best_acc = 0.0
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=device)
            last_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('best_acc', 0.0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print("=> no checkpoint found")
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=False)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict, device=device)
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        utils.save_checkpoint(
            {"state_dict": model.state_dict(),
             "epoch": epoch + 1,
             "best_acc": best_acc,
             "optimizer": optimizer.state_dict(),
             }, None, False, final_output_dir, 'checkpoint_{}.pth'.format(epoch))
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()










