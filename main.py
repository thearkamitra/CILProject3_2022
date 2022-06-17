from pickletools import optimize
from dataset import *
import albumentations as Alb
import sys
import os
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import fcn_resnet50
import pdb
import torch
from utils import train, test
from torch.optim import Adam,lr_scheduler
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from models import FCN_res, Baseline, UNet
import argparse
from loss import GeneralizedDiceLoss
import wandb

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs",type=int, default=20)
    parser.add_argument("-b","--batch",type = int, default=64)
    parser.add_argument("--cmd",type=str, choices=['train','test'],default="train")
    parser.add_argument("--lr",type=float, default=1e-4)
    parser.add_argument("-p","--modeltoload",type=str, default="")
    parser.add_argument("--model",type=str, default="fcn_res", choices = ["fcn_res", "baseline", "unet"])
    parser.add_argument("--modelname",type=str, default="First_check.pth")
    parser.add_argument("--wandb", action='store_true')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RoadCIL("training", training=True, transform=train_transform)
    test_dataset = RoadCIL("test", training=False, transform=test_transform)

    validation_length = int(2*len(dataset)//10)
    train_dataset, validation_dataset = random_split(dataset, [(len(dataset) - validation_length), validation_length])

    batch_size = args.batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model_name = "fcn_res"
    if model_name == "fcn_res":
        model = FCN_res(n_classes = 1)
    if model_name=="baseline":
        model = Baseline(n_classes=1)
    if model_name=="unet":
        model = UNet(in_channel=3,out_channel=1)
    model = model.to(device)
    loss = GeneralizedDiceLoss()
    optimizer = Adam(model.parameters(), args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    if args.cmd=="train":
        if(args.wandb):
            wandb.init(project="cil-project-3", entity="cil-aaaa")
            wandb.config = {"learning_rate": args.lr, "epochs": args.epochs, "batch_size": args.batch}
        train(model, train_dataloader, validation_dataloader,  loss, optimizer, scheduler, device=device, \
              epochs=args.epochs, wandb_log=args.wandb, model_name= args.modelname)
        if(args.wandb):
            wandb.finish()

    if args.cmd=="test":
        if args.modeltoload =="":
            print("No model file selected. Taking the default one.")
            args.modeltoload = "First_check.pth"
        model.load_state_dict(torch.load(args.modeltoload,map_location=torch.device('cpu'))['model_state_dict'])
        model = model.to(device)
        test(model, test_dataloader, device)


if __name__ == "__main__":
    main()