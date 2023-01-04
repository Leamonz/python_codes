import torch
import torch.nn as nn
import albumentations as A
import config
from generator import Generator
from dataset import Pix2PixDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

transform = A.Compose([
    A.Resize(width=256, height=256)
], additional_targets={'image0': 'image'})

test_dataset = Pix2PixDataset('../../dataset/Pix2Pix/data/val', transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
NUM_TEST = 100


def test():
    gen = Generator()
    checkpoint = torch.load('./checkpoints/Generator.pth', map_location=config.DEVICE)
    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    with torch.no_grad():
        count = 1
        for _, (input_image, __) in enumerate(test_loader):
            output = gen(input_image)
            save_image(output * 0.5 + 0.5, r'./results/{:05d}.png'.format(count))
            print("========={:d} Finished!=========".format(count))
            count += 1
            if count == NUM_TEST:
                print("Test Finished!")
                break


if __name__ == '__main__':
    test()
