import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels + 1, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self.block(features_d, features_d * 2, 4, 2, 1),
            self.block(features_d * 2, features_d * 4, 4, 2, 1),
            self.block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat((x, embedding), dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            # 1x1
            self.block(z_dim + embed_size, features_g * 16, 4, 2, 0),
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
        self.embed = nn.Embedding(num_classes, embed_size)

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat((x, embedding), dim=1)
        return self.gen(x)


def Initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
