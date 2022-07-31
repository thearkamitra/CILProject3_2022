import torch

# import torch.nn as nn
# import pdb
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
from sklearn.metrics import roc_curve, RocCurveDisplay, jaccard_score
import matplotlib.pyplot as plt
import cv2
import os

class_labels = {0: "not road", 1: "road"}


def train(
    model,
    train_dataset,
    val_dataset,
    loss,
    optimizer,
    scheduler,
    epochs=4,
    warmup=0,
    model_name="first_check.pth",
    device=torch.device("cpu"),
    wandb_log=False,
    save_path="./",
):
    loss_min = torch.tensor(1e10)  # Min loss for comparison and saving best models
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch} training started.")
        train_epoch(model, train_dataset, optimizer, loss, device, epoch, wandb_log)
        print(f"Epoch {epoch} validation started.")
        is_last_epoch = epoch == epochs - 1
        loss_val = val_epoch(
            model, val_dataset, loss, device, epoch, wandb_log, is_last_epoch
        )
        if loss_val < loss_min:  # Model saved if min val loss obtained
            print("Model weights are saved.")
            loss_min = loss_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "training_loss": loss_val,
                },
                save_path + model_name,
            )
        if epoch > warmup:
            scheduler.step(
                loss_val
            )  # Scheduler changes learning rate based on criterion

    if wandb_log:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        roc_plt = val_plot_auroc(model, val_dataset, device, model_name)
        wandb.log({"ROC": roc_plt, "# Trainable Parameters": num_params})


def train_epoch(model, train_dataset, optimizer, loss_func, device, epoch, wandb_log):
    """
    Training per epoch
    """
    model.train()
    for idx, batch in enumerate(tqdm(train_dataset)):
        optimizer.zero_grad()
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        out = model(images)
        # pdb.set_trace()
        out = torch.sigmoid(out)
        loss = loss_func(out, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()  # Weights are updated
        # get_auroc(out, masks.unsqueeze(1))
        del images, masks, out
        if wandb_log:
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
            wandb.log({"loss": loss, "epoch": epoch, "current lr": current_lr})
    torch.cuda.empty_cache()


def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(
        bg_img,
        masks={
            "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
            "ground truth": {"mask_data": true_mask, "class_labels": class_labels},
        },
    )


def val_epoch(model, val_dataset, loss_func, device, epoch, wandb_log, is_last_epoch):
    """
    Validation Step after each epoch
    """
    model.eval()
    loss_val = 0

    # Setup for IOU curves for different thresholds
    iou_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ious = [0] * len(iou_thresholds)
    iou_dict = dict(zip(iou_thresholds, ious))

    # Setup for IOU curves for different contour removal thresholds
    iou_contours = [150,200,250,300,350,400,450,500]
    cont_ious = [0] * len(iou_contours)
    iou_conts_dict = dict(zip(iou_contours, cont_ious))

    output_road_vals = np.array([])
    output_all_vals = np.array([])
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataset)):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            out = torch.sigmoid(out)
            loss = loss_func(out, masks.unsqueeze(1))
            loss_val += loss / len(val_dataset)
            
            
            print(out.shape)

            # Compute iou
            tar = masks.cpu().numpy().reshape(-1, 1)
            for k, v in iou_dict.items():
                pred = np.where(out.cpu().numpy().reshape(-1, 1) >= k, 1, 0)
                iou_dict[k] += jaccard_score(pred, tar) / len(val_dataset)

            for k,v in iou_conts_dict.items():
                pred = np.where(out.cpu().numpy() >= 0.7, 255, 0).astype(np.uint8)
                cont_removed_pred = np.zeros((0,1))
                for n in range(out.shape[0]):
                    val = (remove_small_contours(pred[n,0], k) / 255.0).reshape(-1,1)
                    #val = (remove_small_contours(pred[n,:], k) / 255.0).reshape(-1,1)
                    cont_removed_pred = np.concatenate((cont_removed_pred, val))
                iou_conts_dict[k] += jaccard_score(cont_removed_pred, tar) / len(val_dataset)

            if is_last_epoch:
                shape = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]
                out_road_vals = (
                    torch.reshape((out * masks.unsqueeze(1)), (1, shape))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                output_all_vals = np.concatenate((output_all_vals, out_road_vals))

                out_road_vals = out_road_vals[out_road_vals.nonzero()]
                output_road_vals = np.concatenate((output_road_vals, out_road_vals))

            if idx == 0 and wandb_log:
                mask_list = []
                heatmap_list = []
                ct = 4 if images.shape[0] >= 4 else images.shape[0]
                outs = torch.round(out[:ct])

                for i in range(ct):
                    pred_mask = torch.reshape(outs[i], (400, 400)).cpu().numpy()
                    mask_list.append(
                        wb_mask(images[i], pred_mask, masks[i].cpu().numpy())
                    )
                    heatmap_list.append(wandb.Image(images[i]))
                    heatmap_list.append(wandb.Image(masks[i].cpu()))
                    heatmap_list.append(wandb.Image(out[i].cpu()))

                wandb.log(
                    {"Predictions": mask_list, "Prediction Heat Maps": heatmap_list}
                )

    if wandb_log:
        validation_iou_log = {}
        for k, v in iou_dict.items():
            validation_iou_log[f"val mIOU, threshold {k}"] = v

        validation_iou_contour_log = {}
        for k, v in iou_conts_dict.items():
            validation_iou_contour_log[f"val mIOU, area {k}"] = v

        wandb.log(validation_iou_log, commit=False)
        wandb.log(validation_iou_contour_log, commit=False)
        wandb.log({"validation loss": loss_val, "epoch": epoch})

        if is_last_epoch:
            wandb.run.summary.update(
                {"final_prediction_roads": wandb.Histogram(output_road_vals)}
            )
            wandb.run.summary.update(
                {"final_prediction_all": wandb.Histogram(output_all_vals)}
            )

    return loss_val


def test(model, test_dataloader, device, method="thres", thres=0.5):
    # Ensure folders are created:
    if not os.path.exists('test/predictions'):
        os.makedirs('test/predictions')
    if not os.path.exists('test/predictions_cont'):
        os.makedirs('test/predictions_cont')

    model.eval()
    with torch.no_grad():
        for _idx, (fname, image) in enumerate(tqdm(test_dataloader)):
            image = image.to(device)
            out_normal = get_output_from_image(model, image).numpy()
            # plt.imsave("test/p1/" + fname[0], round_output(out_normal,method,thres), cmap='gray')

            image_hflip = torch.flip(image,[3])
            out_hflip = get_output_from_image(model, image_hflip)
            out_hflip = torch.flip(out_hflip, [1]).numpy()
            # plt.imsave("test/p2/" + fname[0], round_output(out_hflip,method,thres), cmap='gray')

            image_vflip = torch.flip(image,[2])
            out_vflip = get_output_from_image(model, image_vflip)
            out_vflip = torch.flip(out_vflip, [0]).numpy()
            # plt.imsave("test/p3/" + fname[0], round_output(out_vflip,method,thres), cmap='gray')

            image_rotl = torch.rot90(image, 1, [2,3])
            out_rotl = get_output_from_image(model, image_rotl)
            out_rotl = torch.rot90(out_rotl, -1, [0,1]).numpy()

            image_rotr = torch.rot90(image, -1, [2,3])
            out_rotr = get_output_from_image(model, image_rotr)
            out_rotr = torch.rot90(out_rotr, 1, [0,1]).numpy()

            out = np.mean(np.array([out_normal, out_hflip, out_vflip, out_rotl, out_rotr]), (0))
            out = round_output(out, method, thres)

            int_out = out.astype(np.uint8) * 255
            plt.imsave("test/predictions/" + fname[0], int_out, cmap='gray')
            
            small_contour_removed_out = remove_small_contours(int_out)
            plt.imsave("test/predictions_cont/" + fname[0], small_contour_removed_out, cmap='gray')

def get_output_from_image(model, image):
    score = model(image)
    return torch.reshape(torch.sigmoid(score).cpu(), (400, 400))

def round_output(out, method, thres):
    if method == "thres":
        out_dtype = out.dtype
        out = np.array(out >= thres, dtype=out_dtype)
    elif method == "dynamic":
        thres = np.mean(out)
        out_dtype = out.dtype
        out = np.array(out >= thres, dtype=out_dtype)
    else:
        out = np.around(out)
    return out

def remove_small_contours(img, area=250):
    _ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for c in contours:
        if(cv2.contourArea(c) < area):
            cv2.drawContours(mask, [c], -1, 0, -1)

    new_pred = cv2.bitwise_and(img, img, mask=mask)
    return new_pred


def val_plot_auroc(model, val_dataset, device, name):
    model.eval()
    out_all = np.zeros((0, 1))
    masks_all = np.zeros((0, 1))
    with torch.no_grad():
        for _idx, batch in enumerate(tqdm(val_dataset)):
            images, masks = batch
            images = images.to(device)
            masks = masks.cpu().numpy().reshape(-1, 1)
            out = model(images)
            out = torch.sigmoid(out).cpu().numpy().reshape(-1, 1)
            out_all = np.concatenate((out_all, out))
            masks_all = np.concatenate((masks_all, masks))

    RocCurveDisplay.from_predictions(masks_all, out_all, name=name)
    plt.savefig(name + ".png")
    return plt


def get_auroc(pred, true):
    pred = pred.detach().cpu().numpy().reshape(-1, 1)
    true = true.cpu().numpy().reshape(-1, 1)
    fpr, tpr, thresholds = roc_curve(true, pred)
    return fpr, tpr, thresholds
