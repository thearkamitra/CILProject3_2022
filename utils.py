import torch
import torch.nn as nn
import pdb

def train(model, train_dataset, val_dataset, loss, optimizer, scheduler, \
            epochs=4, model_name = "First_check.pth"):
    loss_val_min = torch.tensor(1e10)
    loss_min = torch.tensor(1e10) #Min loss for comparison and saving best models
    for epoch in range(epochs):    
        print(f"Epoch {epoch} training started.")
        train_epoch(model, train_dataset, optimizer, loss)
        print(f"Epoch {epoch} validation started.")
        loss_val = val_epoch(model, val_dataset, epoch, loss)
        scheduler.step(loss_val) #Scheduler changes learning rate based on criterion
        if loss_val<loss_min: #Model saved if min val loss obtained
            print("Model weights are saved.")
            loss_min = loss_val
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss_val,
            }, model_name)


def train_epoch(model, train_dataset, optimizer, loss_func):
    '''
    Training per epoch
    '''
    for idx, batch in enumerate(train_dataset):
        # pdb.set_trace()
        optimizer.zero_grad()
        images, masks = batch
        out = model(images)
        out = out['out']
        pdb.set_trace()
        loss = loss_func(out, masks.unsqueeze(1))
        loss.backward()
        optimizer.step() #Weights are updated



def val_epoch(model, val_dataset, loss_func):
    '''
    Validation Step after each epoch
    '''
    loss_val = 0
    for idx, batch in enumerate(val_dataset):
        images, masks = batch
        out = model(images)
        loss = loss_func(out, masks)
        loss_val+= loss 
    return loss_val

def test(model, test_dataset,  loss, ):
    pass