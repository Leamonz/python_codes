import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class VGG_net(nn.Module):
    def __init__(self, image_channels, num_features, num_classes, drop_out=0.5):
        super(VGG_net, self).__init__()
        self.features = nn.Sequential(
            Conv(image_channels, num_features),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(num_features, num_features * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(num_features * 2, num_features * 4),
            Conv(num_features * 4, num_features * 4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(num_features * 4, num_features * 8),
            Conv(num_features * 8, num_features * 8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(num_features * 8, num_features * 8),
            Conv(num_features * 8, num_features * 8),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_out),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


IMAGE_CHANNELS = 3
NUM_FEATURES = 64
NUM_CLASSES = 10
DROP_OUT = 0.5
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# mean_rgb=(0.4914, 0.4822, 0.4465),
# std_rgb=(0.2023, 0.1994, 0.2010),
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)])
])
transform2 = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)])
])

model = VGG_net(IMAGE_CHANNELS, NUM_FEATURES, NUM_CLASSES, DROP_OUT).to(DEVICE)
train_dataset = datasets.CIFAR10(root=r'D:\Program_work\PyCharm\Usual\dataset', train=True, download=True,
                                 transform=transform1)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = datasets.CIFAR10(root=r'D:\Program_work\PyCharm\Usual\dataset', train=False, download=True,
                                transform=transform2)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
# weight_decay引入L2正则化
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    running_loss = 0.0
    for _, (image, label) in enumerate(train_loader):
        image, label = image.to(DEVICE), label.to(DEVICE)
        y_pred = model(image)
        loss = criterion(y_pred, label)
        running_loss += loss
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if (_ + 1) % 300 == 0:
            print(f"Epoch: [{epoch + 1}/{NUM_EPOCHS}], LOSS: [{running_loss / 300}]")
            running_loss = 0.0


def test():
    model.eval()
    total = 0
    num_correct = 0
    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.to(DEVICE), label.to(DEVICE)
            y_pred = model(image)
            _, pred_label = torch.max(y_pred, dim=1)
            total += label.shape[0]
            num_correct += (pred_label == label).sum().item()
    print(f"Accuracy: {format(num_correct / total * 100, '.2f')}%")
    model.train()


if __name__ == '__main__':
    for epoch in range(NUM_EPOCHS):
        train(epoch)
        test()
