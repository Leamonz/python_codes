import os
from PIL import Image
import numpy as np

if __name__ == "__main__":
    o_path = r'D:\Program_work\PyCharm\Usual\dataset\CelebA\img_align_celeba'
    t_path = r'D:\Program_work\PyCharm\Usual\dataset\CelebA\train'
    images = os.listdir(o_path)
    len = len(images)
    NUM_IMAGES = 2000
    for i in range(NUM_IMAGES):
        index = int(np.random.rand() * len)
        name = images[index]
        path = o_path + '/' + name
        image = Image.open(path)
        path = t_path + '/{:05d}.png'.format(i + 1)
        image.save(path)
        print(f"{name} moved")
