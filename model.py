import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101
from segmentation_models_pytorch import  Unet
from models import deeplabv3, unet, segformer, resunet


# import os
# import pdb
# import torch
# from utils import train, test
# from torch.optim import Adam,lr_scheduler
# from torch.utils.data import random_split, DataLoader
# import torch.nn.functional as F
# from albumentations.pytorch import ToTensorV2

class Baseline(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(4, 3, 3, padding='valid'),
                                   nn.ReLU(),
                                   nn.Conv2d(3, 32, 3, padding='valid'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, 3, padding='valid'),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 32, 3, padding='valid'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, n_classes, 3, padding='valid'),
                                   )

    def forward(self, x):
        return self.model(x)


class FCN_res(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.model = fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']


class DeepLabv3(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.model = deeplabv3.MTLDeepLabv3({'seg': n_classes})

    def forward(self, x):
        return self.model(x)


class DeepLabv3_Resnet101(nn.Module):

    def __init__(self, n_classes=1):
        super().__init__()
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']


class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # self.model = unet.UNet(in_channel=3, out_channel=n_classes)
        self.model = unet.UNet(num_classes=n_classes)

    def forward(self, x):
        return self.model(x)

class UNetSMP(nn.Module):
    def __init__(self, n_classes=1, decoder_attention=True):
        super().__init__()
        self.model = Unet(encoder_name='resnet101', encoder_weights='imagenet',
                          decoder_attention_type="scse" if decoder_attention else None,
                          in_channels=3,classes=n_classes, encoder_depth=4,
                          decoder_channels=[256, 128, 64, 32])

    def forward(self, x):
        return self.model(x)


class Segformer(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.model = segformer.Segformer(num_classes=n_classes)

    def forward(self, x):
        return self.model(x)


class ResUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.model = resunet.ResUnet(in_channel=3, out_channel=n_classes)

    def forward(self, x):
        return self.model(x)
