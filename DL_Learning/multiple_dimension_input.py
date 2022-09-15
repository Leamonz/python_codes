import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # xy第一个维度的大小即为数据数量
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('datas/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 8D -> 6D
        self.linear2 = torch.nn.Linear(6, 4)  # 6D -> 4D
        self.linear3 = torch.nn.Linear(4, 1)  # 4D -> 1D
        self.activate1 = torch.nn.ReLU()
        self.activate2 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate2(self.linear1(x))
        x = self.activate1(self.linear2(x))
        x = self.activate2(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == "__main__":
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred_val = model(inputs)
            loss = criterion(y_pred_val, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# for epoch in range(1000):
#     y_pred_val = model(x_data)
#     loss = criterion(y_pred_val, y_data)
#     print("Progress:", epoch, format(loss.item(), ".5f"))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
