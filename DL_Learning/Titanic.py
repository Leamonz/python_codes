import numpy as np
import torch
from torch.utils import data


# 1.准备数据
class TitanicDataset(data.Dataset):
    def __init__(self, filepath):
        super(TitanicDataset, self).__init__()
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.len


train_dataset = TitanicDataset('datas/Kaggle-Titanic/train(copy).csv')
train_loader = data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=64)


# test_dataset = TitanicDataset('datas/Kaggle-Titanic/test(copy).csv')
# test_loader = data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=64, num_workers=2)


# 2.建立模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 4)
        self.linear2 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        return x


model = Model()

# 3.构造损失函数和优化算法
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 4.训练
def train(epoch):
    for batch_idx, train_data in enumerate(train_loader, 0):
        features, labels = train_data
        output = model(features)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# def test():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             features, targets = data
#             output = model(features)
#             _, predict = torch.max(output.data, dim=1)
#             total += features.size(0)
#             correct += (predict == targets).sum().item()
#         print(f'Accuracy on test dataset:{format((correct / total) * 100, ".2f")} %')


if __name__ == "__main__":
    for epoch in range(100):
        train(epoch)

target_dataset = np.loadtxt('datas/Kaggle-Titanic/test(copy).csv', delimiter=',', dtype=np.float32)
passenger_id = torch.from_numpy(target_dataset[:, 0])
# target_dataset.dtype = np.float32
inputs = torch.from_numpy(target_dataset[:, 1:])
outputs = model(inputs)
# submission = [passenger_id, outputs]
# submission = torch.cat(submission, dim=1)
# print(submission.data)
