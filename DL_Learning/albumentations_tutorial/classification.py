import cv2
import albumentations as A
import numpy as np
from PIL import Image
from utils import plot_examples

image = Image.open('./images/1_A.jpg')
transform = A.Compose([
    A.Resize(width=1024, height=512),
    A.RandomCrop(width=256, height=256),
    A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=1.0)
])

image_list = [image]
image = np.array(image)

for i in range(15):
    augmentations = transform(image=image)
    augmented_image = augmentations['image']
    image_list.append(augmented_image)

plot_examples(image_list)
