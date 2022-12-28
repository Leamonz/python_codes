import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "./checkpoints/Discriminator.pth"
CHECKPOINT_GEN = "./checkpoints/Generator.pth"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),
     A.HorizontalFlip(p=0.5)],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

transform_only_target = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)
