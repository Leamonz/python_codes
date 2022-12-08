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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 5e-5
BATCH_SIZE = 32
IMAGE_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
NUM_EPOCHS = 10
FEATURE_DISC = 64
FEATURE_GEN = 64
WEIGHT_CLIP = 0.01
NUM_DISC = 5

transform = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
])

dataset = datasets.MNIST(root=r'../../dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
disc = Discriminator(IMG_CHANNELS, FEATURE_DISC).to(device)
gen = Generator(Z_DIM, IMG_CHANNELS, FEATURE_GEN).to(device)
Initialize_weights(disc)
Initialize_weights(gen)

opt_disc = optim.RMSprop(disc.parameters(), lr=lr)
opt_gen = optim.RMSprop(gen.parameters(), lr=lr)

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)
writer_fake = SummaryWriter(r'./runs/WGAN/fake')
writer_real = SummaryWriter(r'./runs/WGAN/real')
step = 0

gen.train()
disc.train()

if __name__ == "__main__":
    if os.path.exists('checkpoints/Discriminator.ckpt'):
        disc.load_state_dict(torch.load('checkpoints/Discriminator.ckpt'))
        print("Discriminator Loaded")
    if os.path.exists('checkpoints/Generator.ckpt'):
        gen.load_state_dict(torch.load('checkpoints/Generator.ckpt'))
        print("Generator Loaded")

    print("============Training Starts============")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            loss_D = 0
            for _ in range(NUM_DISC):
                noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
                fake = gen(noise)
                disc_fake = disc(fake).view(-1)
                disc_real = disc(real).view(-1)
                loss_D = -(torch.mean(disc_real) - torch.mean(disc_fake))
                disc.zero_grad()
                loss_D.backward()
                opt_disc.step()

                for p in disc.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            output = disc(fake).view(-1)
            loss_G = - torch.mean(output)
            gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            if batch_idx == BATCH_SIZE:
                gen.eval()
                disc.eval()
                print(f"Epoch:{epoch + 1}/{NUM_EPOCHS}, lossD:{loss_D:.4f}, lossG:{loss_G:.4f}")

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_fake.add_image("MNIST fake images", img_grid_fake, global_step=step)
                    writer_real.add_image("MNIST real images", img_grid_real, global_step=step)

                step += 1
                torchvision.utils.save_image(img_grid_real, r'./samples/real/{:d}.png'.format(step))
                torchvision.utils.save_image(img_grid_fake, r'./samples/fake/{:d}.png'.format(step))
                torch.save(disc.state_dict(), r'./checkpoints/Discriminator.ckpt')
                torch.save(gen.state_dict(), r'./checkpoints/Generator.ckpt')
                gen.train()
                disc.train()
