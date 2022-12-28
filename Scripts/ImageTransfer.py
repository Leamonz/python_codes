import os
from PIL import Image
import numpy as np
from torchvision import transforms

if __name__ == "__main__":
    resize = transforms.Resize([256, 256])
    o_path = r'D:\Program_work\PyCharm\PaperRepo\dataset\GUOHUA_SHUIMO\test'
    t_path = r'D:\Program_work\PyCharm\PaperRepo\dataset\GUOHUA_SHUIMO\test'
    images = os.listdir(o_path)
    count = 1
    for file in images:
        image_path = os.path.join(o_path, file)
        image = Image.open(image_path)
        image = resize(image)
        target_path = os.path.join(o_path, '{:d}.png'.format(count))
        count += 1
        image.save(target_path)
    print("Transmission Complete")
