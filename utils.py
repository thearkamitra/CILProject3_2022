import torch
# import torch.nn as nn
# import pdb
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.metrics import roc_curve, RocCurveDisplay, jaccard_score
import matplotlib.pyplot as plt

class_labels = {
    0: "not road",
    1: "road"
}


def train(model, train_dataset, val_dataset, loss, optimizer, scheduler,
          epochs=4, warmup=0, model_name="first_check.pth", device=torch.device("cpu"),
          wandb_log=False):
    loss_val_min = torch.tensor(1e10)
    loss_min = torch.tensor(1e10)  # Min loss for comparison and saving best models
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch} training started.")
        train_epoch(model, train_dataset, optimizer, loss, device, wandb_log)
        print(f"Epoch {epoch} validation started.")
        is_last_epoch = epoch == epochs - 1
        loss_val = val_epoch(model, val_dataset, loss, device, wandb_log, is_last_epoch)
        if loss_val < loss_min:  # Model saved if min val loss obtained
            print("Model weights are saved.")
            loss_min = loss_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss_val,
            }, model_name)
        # if epoch > warmup:
        #     scheduler.step(loss_val)  # Scheduler changes learning rate based on criterion
        scheduler.step()
        if wandb_log:
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
            wandb.log({"Current Learning Rate": current_lr})

    if wandb_log:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        roc_plt = val_plot_auroc(model, val_dataset, device, model_name)
        wandb.log({"ROC": roc_plt, "# Trainable Parameters": num_params})


def train_epoch(model, train_dataset, optimizer, loss_func, device, wandb_log):
    """
    Training per epoch
    """
    model.train()
    for idx, batch in enumerate(tqdm(train_dataset)):
        # pdb.set_trace()
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
            wandb.log({"loss": loss})
    torch.cuda.empty_cache()


def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
        "ground truth": {"mask_data": true_mask, "class_labels": class_labels}})


def val_epoch(model, val_dataset, loss_func, device, wandb_log, is_last_epoch):
    """
    Validation Step after each epoch
    """
    model.eval()
    loss_val = 0
    iou_val = 0
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
            loss_val += loss

            # Compute iou
            iou_threshold = 0.5
            p = np.where(out.cpu().numpy().reshape(-1, 1) >= iou_threshold, 1, 0)
            t = masks.cpu().numpy().reshape(-1, 1)
            iou_val += jaccard_score(p, t)

            if is_last_epoch:
                shape = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]
                out_road_vals = torch.reshape((out * masks.unsqueeze(1)), (1, shape)).squeeze(0).cpu().numpy()
                output_all_vals = np.concatenate((output_all_vals, out_road_vals))

                out_road_vals = out_road_vals[out_road_vals.nonzero()]
                output_road_vals = np.concatenate((output_road_vals, out_road_vals))

            if idx == 0 and wandb_log:
                preds_split = []
                mask_list = []
                heatmap_list = []
                ct = 4 if images.shape[0] >= 4 else images.shape[0]
                outs = torch.round(out[:ct])

                for i in range(ct):
                    pred_mask = torch.reshape(outs[i], (400, 400)).cpu().numpy()
                    mask_list.append(wb_mask(images[i], pred_mask, masks[i].cpu().numpy()))
                    preds_split = wandb.Image(np.vstack((images[i], masks[i].cpu().numpy(), pred_mask)), caption="Top: Input, Middle: GT Mask, Bottom: Pred Mask")
                    heatmap_list.append(wandb.Image(images[i]))
                    heatmap_list.append(wandb.Image(out[i].cpu()))

                wandb.log({"Predictions": mask_list})
                wandb.log({"Prediction Heat Maps": heatmap_list})
                wandb.log({"Predictions (split)": preds_split})

    if wandb_log:
        wandb.log({"validation loss": loss_val, "validation mean IOU": iou_val/len(val_dataset)})
        if is_last_epoch:
            wandb.run.summary.update({'final_prediction_roads': wandb.Histogram(output_road_vals)})
            wandb.run.summary.update({'final_prediction_all': wandb.Histogram(output_all_vals)})

    return loss_val


def test(model, test_dataloader, device, method='thres', thres=0.5):
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
