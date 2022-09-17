import torch
import torchvision
from torch.utils import data
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    # 规范化成0-1分布。两个参数分别为均值和标准差
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = torchvision.datasets.FashionMNIST(root='../datas', train=True, transform=transform, download=True)
test_data = torchvision.datasets.FashionMNIST(root='../datas', train=False, transform=transform, download=True)

batch_size = 256
train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=2)
test_loader = data.DataLoader(dataset=test_data, shuffle=False, batch_size=batch_size, num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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


model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)


def train(epoch):
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % 300 == 299:
        #     print(f"epoch:{batch_idx + 1},loss:{format(loss.item(), '.5f')}")


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            outputs = model(inputs)
            _, predict_idx = torch.max(outputs, dim=1)
            correct += (predict_idx == targets).sum().item()
            total += inputs.size(0)
    print(f"Accuracy on test data:{format((correct / (total * 1.0)) * 100, '.5f')}%")


if __name__ == "__main__":
    for epoch in range(50):
        train(epoch)
        test()
