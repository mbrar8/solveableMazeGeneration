import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MazeDataset(Dataset):
    def __init__(self, transform=None):
        self.image_dir = "saved_imgs"
        self.mask_dir = "mask_imgs"
        self.transform = transform

        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        img = np.array(Image.open(img_path).convert('RGB').crop((145, 60, 510, 425)))
        mask = np.array(Image.open(mask_path).convert('RGB').crop((145, 60, 510, 425)))[:,:,0]

        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if mask[i][j] < 150:
                    mask[i][j] = 0
                else:
                    mask[i][j] = 1


        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

