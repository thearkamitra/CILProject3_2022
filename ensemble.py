import torch
from tqdm import tqdm
import numpy as np
import os
import cv2
from torch.utils.data import random_split, DataLoader
from dataset import *

'''Defining fnc from main.py so as to not import (only one fnc needed)'''
test_transform = Alb.Compose(
[
    Alb.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
    ),
    ToTensorV2(),
]
)


def test_without_thres(model, image, device):
    
    model.eval()
    with torch.no_grad():
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

        out = np.mean(np.array([out_normal, out_hflip, out_vflip]), (0))

        return out
            
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
  
  
def ensemble_test(model1_path, model2_path, model3_path, test_dataset, device, method, thres, save_path):

    models = [model1_path, model2_path, model3_path]
    '''Can add more models if needed'''
    
    if not os.path.exists('test/predictions_ensemble'):
        os.makedirs('test/predictions_ensemble')
        
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    for _idx, (fname, image) in enumerate(tqdm(test_dataloader)):
        out_list = []
        for model_path in models:
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            model = model.to(device)
            out = test_without_thres(model, test_dataloader, device)
            out_list.append(out)
        avg_out = np.average(np.array(out_list), axis = 0) #Here can add weights if we think one model better than other (weighted mean)
        final_output = round_output(avg_out, method, thres)
        int_out = final_output.astype(np.uint8) * 255
        plt.imsave(save_path + fname[0], int_out, cmap='gray')
  

'''Run ensembling'''
model_path1 = ''
model_path2 = ''
model_path3 = ''
test_dataset = RoadCIL("insert_test_dataset_path", training=False, transform=test_transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thres = 0.7
save_path = "test/predictions_ensemble/"


ensemble_test(model1_path, model2_path, model3_path, test_dataset, device, "thres", thres, save_path)

