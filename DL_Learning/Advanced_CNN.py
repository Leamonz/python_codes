import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Prepare the data
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    # 规范化成0-1分布。两个参数分别为均值和标准差
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='datas/MNIST/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='datas/MNIST', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# Build the model
# class InceptionA(torch.nn.Module):
#     def __init__(self, in_channels):
#         super(InceptionA, self).__init__()
#         self.avg_pool = torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
#         self.branchavgpool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)
#
#         self.branchpool1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
#
#         self.branchpool5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
#         self.branchpool5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
#
#         self.branchpool3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
#         self.branchpool3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
#         self.branchpool3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         branchpool = self.avg_pool(x)
#         branchpool = self.branchavgpool(branchpool)
#
#         branchpool1x1 = self.branchpool1x1(x)
#
#         branchpool5x5 = self.branchpool5x5_1(x)
#         branchpool5x5 = self.branchpool5x5_2(branchpool5x5)
#
#         branchpool3x3 = self.branchpool3x3_1(x)
#         branchpool3x3 = self.branchpool3x3_2(branchpool3x3)
#         branchpool3x3 = self.branchpool3x3_3(branchpool3x3)
#
#         output = [branchpool, branchpool1x1, branchpool5x5, branchpool3x3]
#         output = torch.cat(output, dim=1)
#         return output
#
#
# class GoogleNet(torch.nn.Module):
#     def __init__(self):
#         super(GoogleNet, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)
#         self.incep1 = InceptionA(10)
#         self.incep2 = InceptionA(20)
#         self.linear1 = torch.nn.Linear(1408, 512)
#         self.linear2 = torch.nn.Linear(512, 256)
#         self.linear3 = torch.nn.Linear(256, 10)
#
#         self.mp = torch.nn.MaxPool2d(kernel_size=2)
#         self.activate = torch.nn.ReLU()
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.mp(self.activate(self.conv1(x)))
#         x = self.incep1(x)
#         x = self.mp(self.activate(self.conv2(x)))
#         x = self.incep2(x)
#         x = x.view(x.size(0), -1)
#         x = self.activate(self.linear1(x))
#         x = self.activate(self.linear2(x))
#         x = self.linear3(x)
#         return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        # 实现一个跳连接 skip connection
        y = self.activate(self.conv(x))
        y = self.conv(y)
        return self.activate(x + y)


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, bias=False)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, bias=False)

        self.activate = torch.nn.ReLU()
        self.mp = torch.nn.MaxPool2d(kernel_size=2)
        self.linear = torch.nn.Linear(512, 10)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

    def forward(self, x):
        x = self.mp(self.activate(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(self.activate(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


model = ResNet()
# 将模块及其参数等迁移到GPU中
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# Build Loss Function and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        # 将输入和输出丢进GPU
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print(f'[{epoch + 1}, {format(batch_idx + 1, "5d")}], Loss:{format(running_loss / 300, ".5f")}')
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # max函数会返回两个值，分别为最大值和最大值的索引，此处只需要其索引
            _, pred_val = torch.max(outputs.data, dim=1)
            total += inputs.size(0)
            correct += (pred_val == targets).sum().item()
    print(f'Accuracy on test dataset:{format((correct / total) * 100, ".2f")} %')


if __name__ == "__main__":
    for epoch in range(100):
        train(epoch)
        test()
