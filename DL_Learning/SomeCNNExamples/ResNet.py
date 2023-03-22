import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.cuda.amp as amp
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm


class ResBlk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, up=False):
        super(ResBlk, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
        )
        if not up:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding)

    def forward(self, x):
        f_x = self.conv(x)
        x = self.projection(x)
        return f_x + x


class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes, residual_blocks=4):
        super(ResNet, self).__init__()

        self.num_residual = residual_blocks
        self.initiate = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_1 = nn.Sequential(
            ResBlk(64, 64, kernel_size=3, stride=1, padding=1),
            ResBlk(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.res_2 = nn.Sequential(
            ResBlk(64, 128, kernel_size=3, stride=1, padding=1),
            ResBlk(128, 128, kernel_size=3, stride=1, padding=1),
        )

        self.res_3 = nn.Sequential(
            ResBlk(128, 256, kernel_size=3, stride=1, padding=1),
            ResBlk(256, 256, kernel_size=3, stride=1, padding=1),
        )

        self.res_4 = nn.Sequential(
            ResBlk(256, 512, kernel_size=3, stride=1, padding=1),
            ResBlk(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(1 * 1 * 512, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.initiate(x)
        x = self.maxpooling(x)
        for i in range(self.num_residual):
            x = getattr(self, 'res_{:d}'.format(i + 1))(x)
            x = self.maxpooling(x)
        x = self.classifier(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                nn.init.normal_(m.weight.data)


# Hyper Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 8
LR = 0.1
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
NUM_EPOCHS = 20
LOG_INTERVAL = 250
SAVE_INTERVAL = 5

# model & optimizer definition
model = ResNet(IMG_CHANNELS, NUM_CLASSES).to(DEVICE)
model.weight_init()
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()
scaler = amp.GradScaler()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5, verbose=True,
                                           threshold=0.5, threshold_mode='abs')

# dataset
train_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]),
])

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]),
])

train_set = datasets.CIFAR10(root='../../dataset/', train=True, transform=train_transform, download=True)
test_set = datasets.CIFAR10(root='../../dataset/', train=False, transform=test_transform, download=True)
eval_set = test_set
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=eval_set, shuffle=False, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=BATCH_SIZE)


def evaluation():
    model.eval()
    num_total = 0
    num_correct = 0
    for _, data in enumerate(val_loader):
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        predictions = model(images)
        _, pred_labels = torch.max(predictions, dim=1)
        num_total += images.shape[0]
        num_correct += (pred_labels == labels).sum().item()
    acc = num_correct / num_total
    model.train()
    return acc


def train():
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.
        for batch_idx, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # with amp.autocast():
            predictions = model(images)
            loss = criterion(predictions, labels)
            running_loss += loss
            optimizer.zero_grad()
            # scaler.scale(loss).backward()
            loss.backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1 == len(train_loader)):
                log_info = f'Epoch: {epoch + 1}/{NUM_EPOCHS} Iteration: {batch_idx + 1}/{len(train_loader)} ' \
                           f'Loss: {running_loss / LOG_INTERVAL}'
                running_loss = 0.0
                logger.info(log_info)
        acc = evaluation()
        log_info = f'Accuracy on epoch {epoch + 1}: {acc * 100}%'
        logger.info(log_info)
        err = 1. - acc
        scheduler.step(err)
        if (epoch + 1) % SAVE_INTERVAL == 0:
            log_info = '===> Saving Model'
            logger.info(log_info)
            torch.save(model.state_dict(), r'ResNet.ckpt')


def test():
    x = torch.randn((1, 3, 224, 224)).to('cuda')
    model = ResNet(3, 10).to('cuda')
    print(model)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    train()
