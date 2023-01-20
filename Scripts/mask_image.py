import os
import argparse

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str, default='D:/program_work/PyCharm/PaperRepo/dataset/DUNHUANG/test')
parser.add_argument('--mask_root', type=str,
                    default='D:/program_work/PyCharm/PaperRepo/dataset/NVIDIA_MASK/mask_20/test')
parser.add_argument('--result_root', type=str, default='./results')

args = parser.parse_args()

if __name__ == "__main__":
    gt_path = args.image_root
    mask_path = args.mask_root

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    gt_folder = os.listdir(gt_path)
    mask_folder = os.listdir(mask_path)
    len1 = len(gt_folder)
    len2 = len(mask_folder)
    assert len1 == len2
    for i in range(len1):
        gt = transform(Image.open(os.path.join(args.image_root, gt_folder[i])))
        mask = transform(Image.open(os.path.join(args.mask_root, mask_folder[i])))

        masked_image = gt * (1 - mask)
        save_image(masked_image, r'{:s}/{:d}.png'.format(args.result_root, i + 1))
        print(i + 1)
