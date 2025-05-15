import torch
import torch.nn as nn
from lib.config import config, update_config
from lib.models.hrnet import HighResolutionNet
from PIL import Image
from torchvision import transforms
import os

def get_hrnet_ad_model(pretrained=True, single_channel=True):
    """Get HRNet model for Alzheimer's Disease classification"""
    model = HighResolutionNet(config)
    
    # Modify first conv layer for single-channel input
    if single_channel:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        if pretrained:
            # Average the weights across RGB channels
            with torch.no_grad():
                model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    
    # Modify final layer for binary classification
    num_channels = sum(config.MODEL.EXTRA.STAGE4.NUM_CHANNELS)
    model.final_layer = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(num_channels, 2)  # 2 classes: AD vs non-AD
    )
    
    return model

def neuroimaging_test(data_path: str, model):
    tsfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = Image.open(data_path).convert("L")  # grayscale
    tensor = tsfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    return bool(logits.argmax(dim=1).item())
