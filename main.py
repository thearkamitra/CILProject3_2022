from dataset import *
import albumentations as Alb
from albumentations.pytorch import ToTensorV2
from utils import train, test, val_plot_auroc
from model import *
import argparse
from loss import *
import wandb
import time
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import random_split, DataLoader

torch.manual_seed(42)

##Define the different transforms for the data
train_transform = Alb.Compose(
    [
        Alb.RandomRotate90(p=0.2),
        Alb.HorizontalFlip(p=0.2),
        Alb.VerticalFlip(p=0.2),
        Alb.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)
##Only normalize the images for testing
test_transform = Alb.Compose(
    [
        Alb.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)


def _parse_args():
    """
    The different arguments for the parser
    """
    parser = argparse.ArgumentParser(
        description="Train and test a model for CIL_2022_Project3"
    )
    parser.add_argument("-e", "--epochs", type=int, default=40)
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-c", "--num_classes", type=int, default=1)
    parser.add_argument(
        "--cmd", type=str, choices=["train", "test", "valauroc"], default="train"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("-p", "--modeltoload", type=str, default="")
    parser.add_argument("-p2", "--model2toload", type=str, default="")
    parser.add_argument("-nnpp", "--nn_post_proc_test_modeltoload", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        default="fcn_res",
        choices=[
            "fcn_res",
            "baseline",
            "unet",
            "deeplabv3",
            "segformer",
            "resunet",
            "deeplabv3_resnet101",
            "unetsmp",
        ],
    )
    parser.add_argument("--model2", type=str, default="fcn_res",
                        choices=["fcn_res", "baseline", "unet", "deeplabv3", "segformer", "resunet", "deeplabv3_resnet101", "unetsmp"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        choices=["dice", "wbce", "wbce2", "bbce", "focal", "tv"],
        default="dice",
    )
    parser.add_argument("-w", "--warmup_steps", type=int, default=0)
    parser.add_argument("-sp", "--save_path", type=str, default="./")
    parser.add_argument(
        "-u",
        "--dataset_to_use",
        type=str,
        choices=["new", "old", "both"],
        default="old",
    )
    return parser.parse_args()


def main():
    """
    Main function for the training and testing of the model
    """
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RoadCIL(
        "massachusetts-road-dataset" if args.pretrain else "training",
        training=True,
        transform=train_transform,
        use=args.dataset_to_use,
    )
    test_dataset = RoadCIL(
        "test", training=False, transform=test_transform, use=args.dataset_to_use
    )
    validation_length = 50 if args.pretrain else int(2 * len(dataset) // 10)
    train_dataset, validation_dataset = random_split(
        dataset, [(len(dataset) - validation_length), validation_length]
    )

    batch_size = args.batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    n_classes = args.num_classes

    if args.pretrain:
        p_dataset = RoadCIL(
            "training", training=True, transform=train_transform, use="new"
        )
        p_validation_length = int(2 * len(p_dataset) // 10)
        p_train_dataset, p_validation_dataset = random_split(
            p_dataset, [(len(p_dataset) - p_validation_length), p_validation_length]
        )
        p_train_dataloader = DataLoader(
            p_train_dataset, batch_size=batch_size, shuffle=True
        )
        p_validation_dataloader = DataLoader(
            p_validation_dataset, batch_size=batch_size, shuffle=False
        )

    # Set Model
    if args.model == "fcn_res":
        model = FCN_res(n_classes)
    if args.model == "baseline":
        model = Baseline(n_classes)
    if args.model == "unet":
        model = UNet(n_classes)
    if args.model == "deeplabv3":
        model = DeepLabv3(n_classes)
    if args.model == "segformer":
        model = Segformer(n_classes)
    if args.model == "resunet":
        model = ResUNet(n_classes)
    if args.model == "deeplabv3_resnet101":
        model = DeepLabv3_Resnet101(n_classes)
    if args.model == "unetsmp":
        model = UNetSMP(n_classes, decoder_attention=False)

    model = model.to(device)
    model2 = FCN_res(n_classes)
    model2 = model2.to(device)

    if args.nn_post_proc_test_modeltoload != "":
        post_processing_model = FCN_res(n_classes, n_layers=101, in_channels=1, pretrained=True)
        post_processing_model = post_processing_model.to(device)
        post_processing_model.load_state_dict(torch.load(args.nn_post_proc_test_modeltoload, map_location=torch.device('cpu'))['model_state_dict'])
    # Load a model for add training, testing or validation
    if args.modeltoload != "":
        model.load_state_dict(
            torch.load(args.modeltoload, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
        )
    
    if args.model2toload != "":
        model2.load_state_dict(
          torch.load(args.model2toload, map_location=torch.device('cpu'))[
            'model_state_dict'
            ]
        )

    # Set Loss
    if args.loss == "gdice":
        loss = GeneralizedDiceLoss()
    elif args.loss == "dice":
        loss = DiceLoss
    elif args.loss == "wbce":
        loss = WeightedBCE
    elif args.loss == "wbce2":
        loss = WeightedBCE2
    elif args.loss == "bbce":
        loss = BorderLossBCE
    elif args.loss == "focal":
        loss = FocalLoss
    elif args.loss == "tv":
        loss = TverskyLoss

    ##Define the optimizer and scheduler
    optimizer = Adam(model.parameters(), args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    run_name = args.model + "-" + args.loss + "-" + time.strftime("%m-%d_%H-%M")

    if args.cmd == "train":
        if args.wandb:
            wandb.init(
                project="cil-project-3",
                entity="cil-aaaa",
                name=run_name,
                group=args.loss,
            )
            wandb.config = {
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch,
            }
        if args.pretrain:
            train(
                model,
                p_train_dataloader,
                p_validation_dataloader,
                loss,
                optimizer,
                scheduler,
                device=device,
                epochs=args.epochs,
                warmup=args.warmup_steps,
                wandb_log=False,
                model_name=run_name + ".pth",
                save_path=args.save_path,
            )
        train(
            model,
            train_dataloader,
            validation_dataloader,
            loss,
            optimizer,
            scheduler,
            device=device,
            epochs=args.epochs,
            warmup=args.warmup_steps,
            wandb_log=args.wandb,
            model_name=run_name + ".pth",
            save_path=args.save_path,
        )
        if args.wandb:
            wandb.finish()

    elif args.cmd == "test":
        model = model.to(device)
        test(model, model2, test_dataloader, device, post_proc=post_processing_model if args.nn_post_proc_test_modeltoload != "" else None)

    elif args.cmd == "valauroc":
        model = model.to(device)
        val_plot_auroc(
            model, validation_dataloader, device=device, name=args.modeltoload
        )


if __name__ == "__main__":
    main()
