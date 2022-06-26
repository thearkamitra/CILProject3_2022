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
from models import *
import argparse
from loss import *
import wandb
import time

train_transform = Alb.Compose(
        [
            Alb.RandomRotate90(p=0.6),
            Alb.HorizontalFlip(p=0.6),
            Alb.VerticalFlip(p=0.6),
            # Alb.ElasticTransform(p=0.5),
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
    parser.add_argument("-e","--epochs",type=int, default=40)
    parser.add_argument("-b","--batch",type = int, default=1)
    parser.add_argument("-c","--num_classes",type = int, default=1)
    parser.add_argument("--cmd",type=str, choices=['train','test'],default="train")
    parser.add_argument("--lr",type=float, default=1e-4)
    parser.add_argument("-p","--modeltoload",type=str, default="")
    parser.add_argument("--model",type=str, default="fcn_res", choices = ["fcn_res", "baseline", "unet","deeplabv3","segformer"])
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("-l","--loss", type=str, choices=["dice", "wbce", "bbce", "focal", "tv"], default="dice")
    parser.add_argument("-w","--warmup_steps",type=int, default=0)
    parser.add_argument("-u", "--dataset_to_use", type=str, choices=["new","old","both"], default="old")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RoadCIL("training", training=True, transform=train_transform, use=args.dataset_to_use)
    test_dataset = RoadCIL("test", training=False, transform=test_transform, use=args.dataset_to_use)
    # pdb.set_trace()
    validation_length = int(2*len(dataset)//10)
    train_dataset, validation_dataset = random_split(dataset, [(len(dataset) - validation_length), validation_length])

    batch_size = args.batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    n_classes = args.num_classes
    # Set Model
    if args.model == "fcn_res":
        model = FCN_res(n_classes)
    if args.model =="baseline":
        model = Baseline(n_classes)
    if args.model =="unet":
        model = UNet(n_classes)
    if args.model=="deeplabv3":
        model = DeepLabv3(n_classes)
    if args.model=="segformer":
        model = Segformer(n_classes)
    model = model.to(device)

    # Set Loss
    if args.loss == 'dice':
        loss = GeneralizedDiceLoss()
    elif args.loss == 'wbce':
        loss = WeightedBCE
    elif args.loss == 'bbce':
        loss = BorderLossBCE
    elif args.loss == 'focal':
        loss = FocalLoss
    elif args.loss == 'tv':
        loss = TverskyLoss
    optimizer = Adam(model.parameters(), args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    run_name = args.model + "-" + args.loss + "-" + time.strftime("%m-%d_%H-%M")

    if args.cmd=="train":
        if(args.wandb):
            wandb.init(project="cil-project-3", entity="cil-aaaa", name=run_name, group=args.loss)
            wandb.config = {"learning_rate": args.lr, "epochs": args.epochs, "batch_size": args.batch}
        train(model, train_dataloader, validation_dataloader,  loss, optimizer, scheduler, device=device, \
              epochs=args.epochs, warmup=args.warmup_steps, wandb_log=args.wandb, model_name= run_name+".pth")
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