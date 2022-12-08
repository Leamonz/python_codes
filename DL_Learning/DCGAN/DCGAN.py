import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self.block(features_d, features_d * 2, 4, 2, 1),
            self.block(features_d * 2, features_d * 4, 4, 2, 1),
            self.block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 1x1
            self.block(z_dim, features_g * 16, 4, 2, 0),
            # 4x4
            self.block(features_g * 16, features_g * 8, 4, 2, 1),
            # 8x8
            self.block(features_g * 8, features_g * 4, 4, 2, 1),
            # 16x16
            self.block(features_g * 4, features_g * 2, 4, 2, 1),
            # 32x32
            nn.ConvTranspose2d(features_g * 2, img_channels, kernel_size=4, stride=2, padding=1),
            # 64x64
            nn.Tanh()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)


def Initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
batch_size = 128
img_size = 64
img_channels = 1
z_dim = 100
num_epochs = 50
feature_d = feature_g = 64
transforms = transforms.Compose([
    transforms.Resize([img_size, img_size]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)]),
])
name = 'MNIST'
dataset = datasets.MNIST(root=r'../../dataset', transform=transforms)
# dataset = datasets.CelebA(root=r'../../dataset', transform=transforms, download=True)
loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
gen = Generator(z_dim, img_channels, feature_g).to(device)
disc = Discriminator(img_channels, feature_d).to(device)
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
writer_fake = SummaryWriter(r'./runs/DCGAN_MNIST/fake')
writer_real = SummaryWriter(r'./runs/DCGAN_MNIST/real')
step = 0
disc_path = r'./checkpoints/Discriminator_MNIST.ckpt'
gen_path = r'./checkpoints/Generator_MNIST.ckpt'

if __name__ == "__main__":
    Initialize_weights(gen)
    Initialize_weights(disc)
    if os.path.exists(disc_path):
        disc.load_state_dict(torch.load(disc_path))
        print("Discriminator loaded")
    if os.path.exists(gen_path):
        gen.load_state_dict(torch.load(gen_path))
        print("Generator loaded")
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)

            # Train Discriminator
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            disc_fake = disc(fake).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

        print(
            f"Epochs: [{epoch + 1}/{num_epochs}], lossD: {lossD:.4f}, lossG: {lossG:.4f}"
        )

        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, img_size, img_size)
            gt = real.reshape(-1, 1, img_size, img_size)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(gt, normalize=True)

            writer_fake.add_image(
                "MNIST Fake Images", img_grid_fake, global_step=step
            )

            writer_real.add_image(
                "MNIST Real Images", img_grid_real, global_step=step
            )
            step += 1
            torchvision.utils.save_image(img_grid_real, r'./runs/%s/real/epoch_%d.png' % (name, step))
            torchvision.utils.save_image(img_grid_fake, r'./runs/%s/fake/epoch_%d.png' % (name, step))
            torch.save(disc.state_dict(), disc_path)
            torch.save(gen.state_dict(), gen_path)
