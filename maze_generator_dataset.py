import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from maze_generator import generate_maze


# Dataset that generates mazes as it goes instead of opening saved images
# Will result in 20x20 resolution instead of 365x365
class MazeGeneratorDataset(Dataset):
    def __init__(self, size, length, mazeMode, transform=None):
        self.size = size
        self.length = length
        self.mazeMode = mazeMode
        self.transform = transform

    def __len__(self):
        return self.length


    def __getitem__(self, index):
        img, mask = generate_maze(1, self.size, self.mazeMode, False)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

