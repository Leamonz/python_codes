import torch
import torch.nn as nn
import torch.optim as optim
import config

from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import Pix2PixDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image


def train(gen, disc, opt_gen, opt_disc, BCE, L1_Loss, train_loader, g_scaler, d_scaler):
    loop = tqdm(train_loader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = L1_Loss(y_fake, y) * config.L1_LAMBDA
            G_loss = L1 + G_fake_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    train_dataset = Pix2PixDataset(root_dir='../../dataset/Pix2Pix/maps/train')
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS)
    val_dataset = Pix2PixDataset(root_dir='../../dataset/Pix2Pix/maps/val')
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=config.BATCH_SIZE)

    for epoch in range(config.NUM_EPOCHS):
        train(gen, disc, opt_gen, opt_disc, BCE, L1_Loss, train_loader, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder='./evaluation')


if __name__ == '__main__':
    main()
