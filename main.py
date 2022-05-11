from pickletools import optimize
from dataset import *
import albumentations as Alb
import os
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import fcn_resnet50
import pdb
from utils import train, test
from torch.optim import Adam
from torch.optim import lr_scheduler
train_transform = Alb.Compose(
        [
            Alb.RandomRotate90(p=0.6),
            Alb.HorizontalFlip(p=0.6),
            Alb.VerticalFlip(p=0.6),
            Alb.ElasticTransform(p=0.5),
            Alb.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
test_transform = Alb.Compose(
        [
            Alb.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )


train_dataset = RoadCIL("training", training=True, transform=train_transform)
test_dataset = RoadCIL("test", training=False, transform=test_transform)

model = fcn_resnet50(pretrained=True)
loss = nn.BCELoss()
optimizer = Adam(model.parameters(), 1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
train(model, train_dataset, loss, optimizer, scheduler)

test(model, test_dataset,  loss, )