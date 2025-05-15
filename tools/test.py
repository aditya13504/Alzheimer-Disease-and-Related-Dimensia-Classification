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
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets.ad_mri import ADMRIDataset
from lib.core import function
from torchvision import datasets


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args

class ToBinary:
    def __init__(self, class_list):
        self.class_list = class_list
        self.ad_classes = [
            "Very Mild Demented", "MildDemented", "ModerateDemented", "Mild Demented", "Moderate Demented"
        ]
    def __call__(self, label_idx):
        classname = self.class_list[label_idx]
        return int(classname in self.ad_classes)

def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # load model
    state_dict = torch.load(args.model_file, map_location=device)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    # Remove head.* keys to avoid size mismatch
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(filtered_state_dict, strict=False)

    # Replace dataset and dataloader with ADMRIDataset
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_ds = datasets.ImageFolder(
        root="./test",
        transform=test_transform
    )
    test_ds.target_transform = ToBinary(test_ds.classes)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    outputs, targets = function.inference(config, test_loader, model, device=device)
    # Example: calculate accuracy
    preds = (torch.sigmoid(outputs) > 0.5).int()
    targets = targets.int()
    accuracy = (preds == targets).float().mean().item()
    print(f'Test Accuracy: {accuracy:.4f}')
    torch.save({'outputs': outputs, 'targets': targets}, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()

