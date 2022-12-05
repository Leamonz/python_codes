import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# build discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


# build generator
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)


# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best default maybe
lr = 3e-4
z_dim = 64
batch_size = 32
# mnist:  28*28*1=784
img_dim = 28 * 28 * 1
num_epochs = 250

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root='../dataset', download=True, transform=transform)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/SimpleGAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/SimpleGAN_MNIST/real")
step = 0

disc_path = r'../checkpoints/Discriminator.ckpt'
gen_path = r'../checkpoints/Generator.ckpt'
if os.path.exists(disc_path):
    disc.load_state_dict(torch.load(disc_path))
    print("Discriminator loaded")
if os.path.exists(gen_path):
    gen.load_state_dict(torch.load(gen_path))
    print("Generator loaded")

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(data_loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        #  Train Discriminator: max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        # 保留计算图，因为后续仍需要使用fake
        lossD.backward(retain_graph=True)
        opt_disc.step()

        #  Train Generator: min log(1-D(G(z))) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epochs: [{epoch + 1}/{num_epochs}], lossD: {lossD:.4f}, lossG: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                gt = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(gt, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )
                step += 1
                torchvision.utils.save_image(img_grid_real, r'./runs/real/%d.png' % step)
                torchvision.utils.save_image(img_grid_fake, r'./runs/fake/%d.png' % step)
                torch.save(disc.state_dict(), disc_path)
                torch.save(gen.state_dict(), gen_path)
