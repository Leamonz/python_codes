import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from options import args

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = args.data_root
TEST_DIR = args.data_root
SAMPLE_PATH = args.sample_path
BATCH_SIZE = args.batch_size
IMG_CHANNELS = args.image_channels
IMG_SIZE = args.image_size
LEARNING_RATE = args.lr
LAMBDA_CYCLE = args.lambda_cycle
NUM_EPOCHS = args.epochs
LOAD_MODEL = args.load_model
SAVE_MODEL = args.save_model
MODEL_PATH = args.model_path
RESULT_PATH = args.result_path
SAMPLE_INTERVAL = args.sample_interval
CHECKPOINT_GEN_M = "GenSummer.pth"
CHECKPOINT_GEN_P = "GenWinter.pth"
CHECKPOINT_DISC_M = "DiscSummer.pth"
CHECKPOINT_DISC_P = "DiscWinter.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

test_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)
