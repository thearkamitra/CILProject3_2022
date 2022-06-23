import os
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import fcn_resnet50
import pdb
import torch
from utils import train, test
from torch.optim import Adam,lr_scheduler
from torch.utils.data import random_split, DataLoader
import deeplabv3
import unet
import segformer
class FCN_res(nn.Module):
    def __init__(self, n_classes = 1) -> None:
        super().__init__()
        self.model = fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size = (1,1), stride = (1,1))
        self.model.aux_classifier[4]= nn.Conv2d(256, n_classes, kernel_size = (1,1), stride = (1,1))
    def forward(self,x):
        return self.model(x)['out']
    

class Baseline(nn.Module):
    def __init__(self, n_classes = 1) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv2d(3, 32, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv2d(64, 32, 3, padding='valid'),
                            nn.ReLU(),
                            nn.Conv2d(32, n_classes, 3, padding='valid'),
                            )
    def forward(self,x):
        return self.model(x)


class DeepLabv3(nn.Module):
    def __init__(self, n_classes = 1) -> None:
        super().__init__()
        self.model = deeplabv3.MTLDeepLabv3({'seg':n_classes})
    def forward(self,x):
        return self.model(x)

class UNet(nn.Module):
    def __init__(self, n_classes = 1) -> None:
        super().__init__()
        self.model = unet.UNet(in_channel = 3, out_channel= n_classes)
    def forward(self,x):
        return self.model(x)


class Segformer(nn.Module):
    def __init__(self, n_classes = 1) -> None:
        super().__init__()
        self.model = segformer.Segformer(num_classes= n_classes)
    def forward(self,x):
        return self.model(x)
