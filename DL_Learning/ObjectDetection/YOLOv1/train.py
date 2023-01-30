import os

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils import *
from loss import YOLOLoss

SEED = 123
torch.manual_seed(SEED)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
PRETRAINED = None
MODEL_PATH = './checkpoint'
IMAGE_DIR = '../../../dataset/PascalVOC/images'
LABEL_DIR = '../../../dataset/PascalVOC/labels'


class Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, bboxes):
        for t in self.transform:
            image, bboxes = t(image), bboxes
        return image, bboxes


transform = Compose([transforms.Resize([448, 448]),
                     transforms.ToTensor()])


def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader)
    mean_loss = []
    for batch_idx, (image, label) in enumerate(loop):
        image, label = image.to(DEVICE), label.to(DEVICE)
        out = model(image)
        loss = loss_fn(out, label)
        mean_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print('[Mean Loss: {:.5f}]'.format(sum(mean_loss) / len(mean_loss)))


def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YOLOLoss()

    if PRETRAINED is not None:
        model_name = os.path.join(MODEL_PATH, '{:d}.pth'.format(PRETRAINED))
        load_checkpoint(torch.load(model_name), model, optimizer)

    train_dataset = VOCDataset(
        "../../../dataset/PascalVOC/100examples.csv",
        transform=transform,
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "../../../dataset/PascalVOC/test.csv", transform=transform, image_dir=IMAGE_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        # if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        train(train_loader, model, optimizer, loss_fn)


if __name__ == '__main__':
    main()
