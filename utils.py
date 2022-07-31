import torch
# import torch.nn as nn
# import pdb
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
from sklearn.metrics import roc_curve, RocCurveDisplay, jaccard_score
import matplotlib.pyplot as plt

class_labels = {
    0: "not road",
    1: "road"
}


def train(model, train_dataset, val_dataset, loss, optimizer, scheduler,
          epochs=4, new_architecture=False, warmup=0, model_name="first_check.pth", device=torch.device("cpu"),
          wandb_log=False, save_path="./"):
    loss_min = torch.tensor(1e10)  # Min loss for comparison and saving best models
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch} training started.")
        train_epoch(model, train_dataset, optimizer, loss, device, epoch, wandb_log, new_architecture)
        print(f"Epoch {epoch} validation started.")
        is_last_epoch = epoch == epochs - 1
        loss_val = val_epoch(model, val_dataset, loss, device, epoch, wandb_log, is_last_epoch, new_architecture)
        if loss_val < loss_min:  # Model saved if min val loss obtained
            print("Model weights are saved.")
            loss_min = loss_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'training_loss': loss_val,
            }, save_path + model_name)
        if epoch > warmup:
            scheduler.step(loss_val)  # Scheduler changes learning rate based on criterion

    if wandb_log:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        roc_plt = val_plot_auroc(model, val_dataset, device, model_name)
        wandb.log({"ROC": roc_plt, "# Trainable Parameters": num_params})


def train_epoch(model, train_dataset, optimizer, loss_func, device, epoch, wandb_log, new_architecture=False):
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
        if new_architecture:
            seg_out = out[:,1:,:,:]
            out = torch.sigmoid(out[:, 0, :, :])
        else:
            out = torch.sigmoid(out)
        loss = loss_func(out, masks.unsqueeze(1))
        
        if new_architecture:
            loss += 0.001*(images- seg_out).pow(2).mean()
        
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
    return wandb.Image(bg_img, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
        "ground truth": {"mask_data": true_mask, "class_labels": class_labels}})


def val_epoch(model, val_dataset, loss_func, device, epoch, wandb_log, new_architecture,  is_last_epoch):
    """
    Validation Step after each epoch
    """
    model.eval()
    loss_val = 0

    iou_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ious = [0] * len(iou_thresholds)
    iou_dict = dict(zip(iou_thresholds, ious))
    output_road_vals = np.array([])
    output_all_vals = np.array([])
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataset)):
            images, masks = batch # if post_process is True, the images will actually be predictions.
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            if new_architecture:
                seg_out = out[:,1:,:,:]
                out = torch.sigmoid(out[:, 0, :, :])
            else:
                out = torch.sigmoid(out)
            loss = loss_func(out, masks.unsqueeze(1))
            
            if new_architecture:
                loss += 0.001*(images- seg_out).pow(2).mean()
            
            loss_val += (loss / len(val_dataset))

            # Compute iou
            tar = masks.cpu().numpy().reshape(-1, 1)
            for k, v in iou_dict.items():
                pred = np.where(out.cpu().numpy().reshape(-1, 1) >= k, 1, 0)
                iou_dict[k] += (jaccard_score(pred, tar) / len(val_dataset))

            if is_last_epoch:
                shape = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]
                out_road_vals = torch.reshape((out * masks.unsqueeze(1)), (1, shape)).squeeze(0).cpu().numpy()
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
                    mask_list.append(wb_mask(images[i], pred_mask, masks[i].cpu().numpy()))
                    heatmap_list.append(wandb.Image(images[i]))
                    heatmap_list.append(wandb.Image(masks[i].cpu()))
                    heatmap_list.append(wandb.Image(out[i].cpu()))

                wandb.log({"Predictions": mask_list, "Prediction Heat Maps": heatmap_list})

    if wandb_log:
        validation_iou_log = {}
        for k, v in iou_dict.items():
            validation_iou_log[f"val mIOU, threshold {k}"] = v

        wandb.log({"validation loss": loss_val, "epoch": epoch})
        wandb.log(validation_iou_log)

        if is_last_epoch:
            wandb.run.summary.update({'final_prediction_roads': wandb.Histogram(output_road_vals)})
            wandb.run.summary.update({'final_prediction_all': wandb.Histogram(output_all_vals)})

    return loss_val


def test(model, test_dataloader, device, pred_transform=None, method='thres', thres=0.7, post_model=None):
    model.eval()
    with torch.no_grad():
        for _idx, (fname, image) in enumerate(tqdm(test_dataloader)):
            image = image.to(device)
            score = model(image)
            out = torch.reshape(torch.sigmoid(score).cpu(), (400, 400)).numpy()
            if method == "thres":
                out_dtype = out.dtype
                out = np.array(out >= thres, dtype=out_dtype)
            elif method == "dynamic":
                thres = np.mean(out)
                out_dtype = out.dtype
                out = np.array(out >= thres, dtype=out_dtype)
            else:
                out = np.around(out)
            if post_model is not None:
                pred = pred_transform(image=out)["image"].unsqueeze(0)
                pred = pred.to(device)
                score_pp = post_model(pred)
                out_pp = torch.reshape(torch.sigmoid(score_pp).cpu(), (400, 400)).numpy()
                final = np.array(out_pp >= thres, dtype=out.dtype)
                plt.imsave('test/predictions_post_processed/' + fname[0], final)
            plt.imsave('test/predictions/' + fname[0], out)


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
    plt.savefig(name + '.png')
    return plt


def get_auroc(pred, true):
    pred = pred.detach().cpu().numpy().reshape(-1, 1)
    true = true.cpu().numpy().reshape(-1, 1)
    fpr, tpr, thresholds = roc_curve(true, pred)
    return fpr, tpr, thresholds
