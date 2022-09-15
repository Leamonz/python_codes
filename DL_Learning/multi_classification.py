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
class MultiClassification(torch.nn.Module):
    def __init__(self):
        super(MultiClassification, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = x.view((-1, 784))
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        x = self.linear5(x)
        return x


model = MultiClassification()
# Build Loss Function and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
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
