from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config


class Pix2PixDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.list_files[index]))
        image = np.asarray(image)
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]
        # Augmentations
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']
        input_image = config.transform_only_input(image=input_image)['image']
        target_image = config.transform_only_target(image=target_image)['image']

        return input_image, target_image
