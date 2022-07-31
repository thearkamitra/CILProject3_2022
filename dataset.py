import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch


class RoadCIL(Dataset):
    def __init__(self, img_dir, mask_dir=None, training=False, transform=None, use="old"):
        self.img_dir = img_dir + "/images/"
        if mask_dir is None:
            self.mask_dir = img_dir
        self.mask_dir = self.mask_dir + "/groundtruth/"
        self.training = training
        self.transform = transform
        self.images = os.listdir(self.img_dir)
        # if not training:
        #     use = "both"
        # self.use = use
        # if use == "old":
        #     self.images = [x for x in self.images if ".tiff" not in x]
        # elif use == "new":
        #     self.images = [x for x in self.images if ".tiff" in x]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.training is False:
            # return test file names and image set
            if self.transform is not None:
                augmentation = self.transform(image=image)
                image = augmentation["image"]
            return self.images[index], image
        else:
            mask_path = os.path.join(self.mask_dir, self.images[index])
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
            if self.transform is not None:
                augmentation = self.transform(image=image, mask=mask)
                image = augmentation["image"]
                mask = augmentation["mask"]
            return image, mask

class RoadCILPostProcess(Dataset):
    def __init__(self, img_dir, segmentation_model, segmentation_model2, segmentation_transform, pred_transform, device, thresh=0.7, mask_dir=None, training=False):
        self.img_dir = img_dir + "/images/"
        if mask_dir is None:
            self.mask_dir = img_dir
        self.mask_dir = self.mask_dir + "/groundtruth/"
        self.training = training
        self.pred_transform = pred_transform
        self.images = os.listdir(self.img_dir)
        self.model = segmentation_model
        self.model2 = segmentation_model2
        self.thresh = thresh
        self.device = device
        self.seg_transform = segmentation_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        self.model.eval()
        with torch.no_grad():
            augmentation = self.seg_transform(image=image)
            image = augmentation["image"].unsqueeze(0)
            image = image.to(self.device)
            score1 = self.model(image.clone())
            score2 = self.model2(image.clone())

            seg_pred1 = torch.reshape(torch.sigmoid(score1).cpu(), (400, 400)).numpy()
            seg_pred2 = torch.reshape(torch.sigmoid(score2).cpu(), (400, 400)).numpy()
            seg_pred = np.mean(np.array([seg_pred1, seg_pred2]), (0))
            seg_pred = np.array(seg_pred >= self.thresh, dtype=seg_pred.dtype)

        if self.training is False:
            # return test file names and image set
            if self.pred_transform is not None:
                augmentation = self.pred_transform(image=seg_pred)
                seg_pred = augmentation["image"]
            return self.images[index], seg_pred
        else:
            mask_path = os.path.join(self.mask_dir, self.images[index])
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
            if self.pred_transform is not None:
                augmentation = self.pred_transform(image=seg_pred, mask=mask)
                seg_pred = augmentation["image"]
                mask = augmentation["mask"]
            return seg_pred, mask