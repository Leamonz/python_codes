# import cv2
# from PIL import Image
# import numpy as np
# import os
# import random
#
# t_path = './results'
# num_masks = 10
# num_walks = 10000
#
# if __name__ == '__main__':
#     for i in range(num_masks):
#         mask = np.full((600, 600, 3), 0, dtype=np.uint8)
#         for _ in range(num_walks):
#             x = random.randint(0, 599)
#             y = random.randint(0, 599)
#             mask[x, y, :] = 255
#         print(mask)
#         # print(mask)
#         # mask = Image.fromarray(mask)
#         # mask.save(r'{:s}/{:d}.png'.format(t_path, i + 1))
#         cv2.imwrite(r'{:s}/{:d}.png'.format(t_path, i + 1), mask)
import PIL.Image
import numpy as np
import torch
import torchvision.transforms
from torchvision.utils import save_image
from skimage.feature import canny
from scipy.misc import imread
from skimage.color import rgb2gray, gray2rgb

if __name__ == '__main__':
    resize = torchvision.transforms.Resize([256, 256])
    img = imread('./32.png')
    img = resize(PIL.Image.fromarray(img))
    img = np.asarray(img)
    mask = np.asarray(resize(PIL.Image.fromarray(imread('./mask.png'))))
    img_gray = rgb2gray(img)
    edge = canny(img_gray, sigma=2, mask=mask).astype(np.float)
    save_image(torch.from_numpy(edge), 'edge.png')
