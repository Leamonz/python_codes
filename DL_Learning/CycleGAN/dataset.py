import torch
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
import config
import numpy as np


class MonetPhotoDataset(Dataset):
    def __init__(self, root_Monet, root_Photo, transforms=None):
        super(MonetPhotoDataset, self).__init__()
        self.root_Monet = root_Monet
        self.root_Photo = root_Photo
        self.transform = transforms

        self.MonetImages = os.listdir(self.root_Monet)
        self.PhotoImages = os.listdir(self.root_Photo)
        self.len = max(len(self.MonetImages), len(self.PhotoImages))
        self.Monet_len = len(self.MonetImages)
        self.Photo_len = len(self.PhotoImages)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        monet = self.MonetImages[index % self.Monet_len]
        photo = self.PhotoImages[index % self.Photo_len]
        monet_path = self.root_Monet + '/' + monet
        photo_path = self.root_Photo + '/' + photo
        monet, photo = Image.open(monet_path).convert('RGB'), Image.open(photo_path).convert('RGB')
        monet, photo = np.asarray(monet), np.asarray(photo)

        if self.transform:
            augs = self.transform(image=monet, image0=photo)
            monet, photo = augs['image'], augs['image0']

        return monet, photo
