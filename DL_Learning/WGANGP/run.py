import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import Initialize_weights
from model import Discriminator
from model import Generator
from utils import gradient_penalty
from dataset import WGANDataset
from time import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 64
IMG_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 10
FEATURE_DISC = 64
FEATURE_GEN = 64
NUM_DISC = 5
LAMBDA_GP = 10
MODE = 'train'

transform = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
])

dataset = WGANDataset('../../dataset/CelebA/train', transform)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
disc = Discriminator(IMG_CHANNELS, FEATURE_DISC).to(device)
gen = Generator(Z_DIM, IMG_CHANNELS, FEATURE_GEN).to(device)
Initialize_weights(disc)
Initialize_weights(gen)

opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.0, 0.9))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))

fixed_noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)


def train():
    gen.train()
    disc.train()
    if os.path.exists('./checkpoints/Discriminator.ckpt'):
        disc.load_state_dict(torch.load('./checkpoints/Discriminator.ckpt'))
        print("==> Loading Discriminator")
    if os.path.exists('./checkpoints/Generator.ckpt'):
        gen.load_state_dict(torch.load('./checkpoints/Generator.ckpt'))
        print("==> Loading Generator")

    count: int = 0
    print("============Training Starts============")
    for epoch in range(NUM_EPOCHS):
        start = time()
        for batch_idx, real in enumerate(loader):
            real = real.to(device)
            loss_D = 0
            batch_size = real.shape[0]
            for _ in range(NUM_DISC):
                noise = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
                fake = gen(noise)
                disc_fake = disc(fake).view(-1)
                disc_real = disc(real).view(-1)
                gp = gradient_penalty(disc, real, fake, device)
                loss_D = (-(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp)
                disc.zero_grad()
                loss_D.backward()
                opt_disc.step()

            noise = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            output = disc(fake).view(-1)
            loss_G = - torch.mean(output)
            gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            if batch_idx == 0:
                gen.eval()
                disc.eval()
                print(f"[Epoch:{epoch + 1}/{NUM_EPOCHS}], [lossD:{loss_D:.4f}], [lossG:{loss_G:.4f}]")

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(real, normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                    torchvision.utils.save_image(img_grid_real,
                                                 "{:s}/epoch{:d}_real{:d}.png".format('./samples/real', epoch + 1,
                                                                                      count + 1))
                    torchvision.utils.save_image(img_grid_fake,
                                                 "{:s}/epoch{:d}_fake{:d}.png".format('./samples/fake', epoch + 1,
                                                                                      count + 1))

                    gen.train()
                    disc.train()
        end = time()
        print(f"Time taken: {(end - start) * 1000:.2f}s")
        torch.save(disc.state_dict(), r'./checkpoints/Discriminator.ckpt')
        torch.save(gen.state_dict(), r'./checkpoints/Generator.ckpt')


def test():
    disc.load_state_dict(torch.load('./checkpoints/Discriminator.ckpt'))
    gen.load_state_dict(torch.load('./checkpoints/Generator.ckpt'))
    disc.eval()
    gen.eval()
    random_noise = torch.randn(IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    fake = gen(random_noise)
    print("Image Generated!")
    torchvision.utils.save_image(fake, './results/GeneratedImage')


if __name__ == '__main__':
    if MODE == 'train':
        train()
    else:
        test()
