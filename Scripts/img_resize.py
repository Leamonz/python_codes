import argparse
import os
import torchvision.transforms as transforms
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default=None, help="The source image directory.")
parser.add_argument('--output', type=str, default=None, help="The output directory.")
parser.add_argument('--size', type=str, default='32x32', help="The size of target image.(HxW)")

args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.input):
        print("Source directory doesn;t exist!")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    size = args.size.split("x")
    resize = transforms.Resize([int(size[0]), int(size[1])])

    src = os.listdir(args.input)
    for file in src:
        print(f"\'{file}\' size changed")
        o_path = args.input + '/' + file
        image = Image.open(o_path)
        image = resize(image)
        t_path = args.output + '/' + file
        image.save(t_path)
