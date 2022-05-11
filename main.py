from pickletools import optimize
from dataset import *
import albumentations as Alb
import os
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import fcn_resnet50
import pdb
from utils import train, test
from torch.optim import Adam,lr_scheduler
from torch.utils.data import random_split, DataLoader
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


dataset = RoadCIL("training", training=True, transform=train_transform)
test_dataset = RoadCIL("test", training=False, transform=test_transform)

validation_length = int(2*len(dataset)//10)
train_dataset, validation_dataset = random_split(dataset, [(len(dataset) - validation_length), validation_length])

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model_name = "fcn_res"
if model_name == "fcn_res":
    model = fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size = (1,1), stride = (1,1))
    model.aux_classifier[4]= nn.Conv2d(256, 1, kernel_size = (1,1), stride = (1,1))
loss = nn.BCELoss()
optimizer = Adam(model.parameters(), 1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
train(model, train_dataloader, validation_dataloader,  loss, optimizer, scheduler)

test(model, test_dataset,  loss, )