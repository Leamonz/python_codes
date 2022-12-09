import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "../../dataset/CycleGAN/monet2photo/train"
TEST_DIR = "../../dataset/CycleGAN/monet2photo/test"
SAMPLE_PATH = "./samples"
BATCH_SIZE = 1
IMG_CHANNELS = 3
IMG_SIZE = 256
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0
NUM_EPOCHS = 20
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH = "./models"
CHECKPOINT_GEN_M = "GenMonet.pth"
CHECKPOINT_GEN_P = "GenPhoto.pth"
CHECKPOINT_DISC_M = "DiscMonet.pth"
CHECKPOINT_DISC_P = "DiscPhoto.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
