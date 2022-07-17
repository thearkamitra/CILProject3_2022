import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class RoadCIL(Dataset):
    def __init__(self, img_dir, mask_dir=None, training=False, transform=None, use= "old"):
        self.img_dir = img_dir+"/images/"
        if mask_dir is None:
            self.mask_dir = img_dir
        self.mask_dir = self.mask_dir+"/groundtruth/"
        self.training = training
        self.transform = transform
        self.images = os.listdir(self.img_dir)
        if not training:
            use = "both"
        self.use = use
        if use=="old":
            self.images = [x for x in self.images if ".tiff" not in x]
        elif use=="new":
            self.images = [x for x in self.images if ".tiff" in x]

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
