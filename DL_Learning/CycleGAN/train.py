import torch
import torchvision.utils

from dataset import MonetPhotoDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from discriminator import Discriminator
from generator import Generator


def train(disc_M, disc_P, gen_M, gen_P, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler):
    count: int = 1
    for batch_idx, (monet, photo) in enumerate(loader):
        monet, photo = monet.to(config.DEVICE), photo.to(config.DEVICE)

        # 训练Discriminator
        with torch.cuda.amp.autocast():
            fake_photo = gen_P(monet)
            D_Photo_Real = disc_P(photo)
            D_Photo_Fake = disc_P(fake_photo.detach())
            loss_D_Photo_Real = MSE(D_Photo_Real, torch.ones_like(D_Photo_Real))
            loss_D_Photo_Fake = MSE(D_Photo_Fake, torch.zeros_like(D_Photo_Fake))
            loss_D_Photo = loss_D_Photo_Fake + loss_D_Photo_Real

            fake_monet = gen_M(photo)
            D_Monet_Real = disc_M(monet)
            D_Monet_Fake = disc_M(fake_monet.detach())
            loss_D_Monet_Real = MSE(D_Monet_Real, torch.ones_like(D_Monet_Real))
            loss_D_Monet_Fake = MSE(D_Monet_Fake, torch.ones_like(D_Monet_Fake))
            loss_D_Monet = loss_D_Monet_Fake + loss_D_Monet_Real

            loss_D = (loss_D_Monet + loss_D_Photo) / 2

        opt_disc.zero_grad()
        d_scaler.scale(loss_D).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 训练Generator
        with torch.cuda.amp.autocast():
            D_P_Fake = disc_P(fake_photo)
            D_M_Fake = disc_M(fake_monet)
            loss_G_Photo = MSE(D_P_Fake, torch.ones_like(D_P_Fake))
            loss_G_Monet = MSE(D_M_Fake, torch.ones_like(D_M_Fake))

            # cycle consistency loss & identity loss
            cycle_Monet = gen_M(fake_photo)
            cycle_Photo = gen_P(fake_monet)
            cycle_monet_loss = L1(monet, cycle_Monet)
            cycle_photo_loss = L1(photo, cycle_Photo)

            loss_G = loss_G_Monet \
                     + config.LAMBDA_CYCLE * cycle_monet_loss \
                     + loss_G_Photo \
                     + config.LAMBDA_CYCLE * cycle_photo_loss

        opt_gen.zero_grad()
        g_scaler.scale(loss_G).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if batch_idx % 200 == 0:
            print(f"[iterations:{batch_idx}/{len(loader)}], [loss_D:{loss_D.item()}], [loss_G:{loss_G.item()}]")
            # Sample Images
            torchvision.utils.save_image(fake_monet * 0.5 + 0.5,
                                         r'{:s}/monet_sample_{:d}.png'.format(config.SAMPLE_PATH, count))
            torchvision.utils.save_image(fake_photo * 0.5 + 0.5,
                                         r'{:s}/photo_sample_{:d}.png'.format(config.SAMPLE_PATH, count))
            count += 1


def main():
    disc_M = Discriminator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    disc_P = Discriminator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    gen_M = Generator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    gen_P = Generator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    opt_disc = optim.Adam(list(disc_P.parameters()) + list(disc_M.parameters()),
                          lr=config.LEARNING_RATE,
                          betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_P.parameters()) + list(gen_M.parameters()),
                         lr=config.LEARNING_RATE,
                         betas=(0.5, 0.999))
    # cycle consistency loss & identity loss
    L1 = nn.L1Loss()
    # generative adversarial loss
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_M, gen_M, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_M, disc_M, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_P, disc_P, opt_disc, config.LEARNING_RATE)
    dataset = MonetPhotoDataset(root_Monet=config.TRAIN_DIR + '/monet',
                                root_Photo=config.TRAIN_DIR + '/photo',
                                transforms=config.transforms)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS):
        print(f"=========Epoch{epoch + 1} Starts!=========")
        train(disc_M, disc_P, gen_M, gen_P, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(disc_M, opt_disc, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_M)
            save_checkpoint(disc_P, opt_disc, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_P)
            save_checkpoint(gen_M, opt_gen, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_M)
            save_checkpoint(gen_P, opt_gen, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_P)
        print(f"=========Epoch{epoch + 1} Ends!=========")


if __name__ == "__main__":
    main()
