import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class WGANDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(WGANDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.images = os.listdir(self.root)

        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = self.images[index]
        path = self.root + '/' + img
        real = Image.open(path)

        if transforms is not None:
            real = self.transforms(real)

        return real
