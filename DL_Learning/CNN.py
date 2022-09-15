# import torch
#
# # view(Batch, Channel, Width_in, Height_in)
# # input = torch.tensor([3, 4, 6, 5, 7,
# #                       2, 4, 6, 8, 2,
# #                       1, 6, 7, 8, 4,
# #                       9, 7, 4, 6, 2,
# #                       3, 7, 5, 4, 1]).view(1, 1, 5, 5)
# # Conv2d(InChannels, OutChannels)
# # padding---填充
# # conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
# # 输出结果会是1*1*5*5
# # strid---步长  两次卷积间，卷积核中心的距离
# # conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
# # 输出结果会是1*1*2*2
#
# # conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False)
# # # view(Output_Channel, Input_Channel, Kernel_Width, Kernel_Height)
# # kernel = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
# # conv_layer.weight.data = kernel.data
# #
# # with torch.no_grad():
# #     output = conv_layer(input)
# #     print(output.data)
#
# input = torch.tensor([3, 4, 6, 5,
#                       2, 4, 6, 8,
#                       1, 6, 7, 8,
#                       9, 7, 4, 6], dtype=torch.float32).view(1, 1, 4, 4)
#
# # 最大池化层
# maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
# output = maxpooling_layer(input)
# print(output)

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
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.linear1 = torch.nn.Linear(320, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 10)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(self.activate(self.conv1(x)))
        x = self.pooling(self.activate(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.linear4(x)
        return x


model = SimpleCNN()
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
