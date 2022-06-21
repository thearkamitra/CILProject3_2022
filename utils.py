import torch
import torch.nn as nn
import pdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb

class_labels = {
  0: "not road",
  1: "road"
}

def train(model, train_dataset, val_dataset, loss, optimizer, scheduler, \
            epochs=4, model_name = "First_check.pth", device = torch.device("cpu"), \
            wandb_log=False):
    loss_val_min = torch.tensor(1e10)
    loss_min = torch.tensor(1e10) #Min loss for comparison and saving best models
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch} training started.")
        train_epoch(model, train_dataset, optimizer, loss, device, wandb_log)
        print(f"Epoch {epoch} validation started.")
        loss_val = val_epoch(model, val_dataset, loss, device, wandb_log)
        scheduler.step(loss_val) #Scheduler changes learning rate based on criterion
        if loss_val<loss_min: #Model saved if min val loss obtained
            print("Model weights are saved.")
            loss_min = loss_val
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss_val,
            }, model_name)


def train_epoch(model, train_dataset, optimizer, loss_func, device, wandb_log):
    '''
    Training per epoch
    '''
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
        optimizer.step() #Weights are updated
        del images, masks, out
        if(wandb_log):
            wandb.log({"loss": loss})
    torch.cuda.empty_cache()

def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
        "prediction" : {"mask_data" : pred_mask, "class_labels" : class_labels},
        "ground truth" : {"mask_data" : true_mask, "class_labels" : class_labels}})

def val_epoch(model, val_dataset, loss_func, device, wandb_log):
    '''
    Validation Step after each epoch
    '''
    model.eval()
    loss_val = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataset)):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            out = torch.sigmoid(out)
            loss = loss_func(out, masks.unsqueeze(1))
            loss_val+= loss

            if (idx==0 and wandb_log):
                mask_list = []
                ct = 4 if images.shape[0] >= 4 else images.shape[0]
                outs = torch.round(out[:ct])

                for i in range(ct):
                    pred_mask = torch.reshape(outs[i], (400,400)).numpy()
                    mask_list.append(wb_mask(images[i], pred_mask, masks[i].numpy()))

                wandb.log({"Predictions": mask_list})
    
    if(wandb_log):
        wandb.log({"validation loss": loss_val})

    return loss_val

def test(model, test_dataloader, device, method='thres', thres = 0.5):
    model.eval()
    with torch.no_grad():
        for _idx,(fname,image) in enumerate(tqdm(test_dataloader)):
            image = image.to(device)
            score = model(image)
            out = torch.reshape(torch.sigmoid(score).cpu(), (400,400)).numpy()
            if method=="thres":
                out_dtype = out.dtype
                out = np.array(out>=thres, dtype=out_dtype) 
            elif method =="dynamic":
                thres = np.mean(out)
                out_dtype = out.dtype
                out = np.array(out>=thres, dtype=out_dtype) 
            else:
                out = np.around(out)
            plt.imsave('test/predictions/' + fname[0], out)