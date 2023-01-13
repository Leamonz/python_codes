import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Inception(nn.Module):
    def __init__(self, input_channels, output_channels, features):
        super(Inception, self).__init__()
        output = features[0] + features[2] + features[4] + features[5]
        # 检测输出通道数是否正确
        assert output_channels == output

        self.path1 = nn.Sequential(
            nn.Conv2d(input_channels, features[0], kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(input_channels, features[1], kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], features[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(input_channels, features[3], kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[3], features[4], kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(input_channels, features[5], kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class GoogleNet(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(GoogleNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.side1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 4 * 128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes)
        )
        self.side2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(528, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 4 * 128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes)
        )
        self.Inception3a = Inception(192, 256, [64, 96, 128, 16, 32, 32])
        self.Inception3b = Inception(256, 480, [128, 128, 192, 32, 96, 64])
        self.Inception4a = Inception(480, 512, [192, 96, 208, 16, 48, 64])
        self.Inception4b = Inception(512, 512, [160, 112, 224, 24, 64, 64])
        self.Inception4c = Inception(512, 512, [128, 128, 256, 24, 64, 64])
        self.Inception4d = Inception(512, 528, [112, 144, 288, 32, 64, 64])
        self.Inception4e = Inception(528, 832, [256, 160, 320, 32, 128, 128])
        self.Inception5a = Inception(832, 832, [256, 160, 320, 32, 128, 128])
        self.Inception5b = Inception(832, 1024, [384, 192, 384, 48, 128, 128])
        self.final_classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1 * 1 * 1024, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.maxpool(x)
        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.maxpool(x)
        x = self.Inception4a(x)
        output1 = self.side1(x)
        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)
        output2 = self.side2(x)
        x = self.Inception4e(x)
        x = self.maxpool(x)
        x = self.Inception5a(x)
        x = self.Inception5b(x)
        output3 = self.final_classifier(x)
        return output1, output2, output3


BATCH_SIZE = 32
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
LAMBDA_SIDE_LOSS = 0.3
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
NUM_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize([229, 229]), transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMAGE_CHANNELS)],
                         [0.5 for _ in range(IMAGE_CHANNELS)])
])
transform2 = transforms.Compose([
    transforms.Resize([229, 229]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMAGE_CHANNELS)],
                         [0.5 for _ in range(IMAGE_CHANNELS)])
])

model = GoogleNet(IMAGE_CHANNELS, NUM_CLASSES).to(DEVICE)
train_dataset = datasets.CIFAR10(root=r'D:\Program_work\PyCharm\Usual\dataset', train=True, download=True,
                                 transform=transform1)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = datasets.CIFAR10(root=r'D:\Program_work\PyCharm\Usual\dataset', train=False, download=True,
                                transform=transform2)
test_loader = DataLoader(test_dataset, shuffle=False,
                         batch_size=BATCH_SIZE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler1 = StepLR(optimizer, step_size=8, gamma=0.96, verbose=True)
scheduler2 = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, threshold=0.5, threshold_mode='abs', verbose=True,
                               patience=2, min_lr=1e-5)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    running_loss = 0.0
    for _, (image, label) in enumerate(train_loader):
        image, label = image.to(DEVICE), label.to(DEVICE)
        pred1, pred2, pred3 = model(image)
        loss = (criterion(pred1, label) + criterion(pred2, label)) * LAMBDA_SIDE_LOSS + criterion(pred3, label)
        running_loss += loss
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if (_ + 1) % 500 == 0:
            print(f"Epochs: [{epoch + 1}/{NUM_EPOCHS}], Loss:[{running_loss / 500}]")
            running_loss = 0.0


def test():
    model.eval()
    total = 0
    num_correct = 0
    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.to(DEVICE), label.to(DEVICE)
            __, ___, output = model(image)
            _, pred_label = torch.max(output, dim=1)
            total += label.shape[0]
            num_correct += (pred_label == label).sum().item()
    acc = num_correct / total * 100
    print("Accuracy: {:.2f}%".format(acc))
    model.train()
    return acc


if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        train(epoch)
        accuracy = test()
        scheduler1.step()
        scheduler2.step(accuracy)
