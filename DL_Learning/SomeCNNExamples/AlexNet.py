import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


class AlexNet(nn.Module):
    def __init__(self, image_channels, num_classes, drop_out=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgPool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_out),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgPool(x)
        x = self.flatten(x)
        return self.classifier(x)


IMG_CHANNELS = 3
NUM_CLASSES = 100
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
DROP_OUT = 0.5
NUM_EPOCHS = 50
BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize([227, 227]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
])
transform2 = transforms.Compose([
    transforms.Resize([227, 227]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
])

model = AlexNet(IMG_CHANNELS, NUM_CLASSES, DROP_OUT).to(device)
train_dataset = datasets.CIFAR10(root=r'D:\Program_work\PyCharm\Usual\dataset', train=True, download=True,
                                 transform=transform1)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = datasets.CIFAR10(root=r'D:\Program_work\PyCharm\Usual\dataset', train=False, download=True,
                                transform=transform2)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    running_loss = 0.0
    for _, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        y_pred = model(image)
        loss = criterion(y_pred, label)
        running_loss += loss
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if (_ + 1) % 256 == 0:
            print(f"Epochs: [{epoch + 1}/{NUM_EPOCHS}], LOSS: [{running_loss / 256}]")
            running_loss = 0.0


def test():
    model.eval()
    total = 0
    num_correct = 0
    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            y_pred = model(image)
            _, pred_label = torch.max(y_pred, dim=1)
            total += label.shape[0]
            num_correct += (pred_label == label).sum().item()
    print(f"Accuracy: {format(num_correct / total * 100, '.2f')}%")
    model.train()


if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        train(epoch)
        test()
