import torch
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0], [0], [1.0]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 利用sigmoid函数将函数值转变为0-1的值
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()
# 交叉熵损失函数
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(1000):
    y_pred_val = model(x_data)
    loss = criterion(y_pred_val, y_data)
    print("Progress:", epoch, format(loss.item(), ".5f"))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w:", model.linear.weight.item(), "b:", model.linear.bias.item())

x = np.linspace(0, 10, 200, dtype='float32')
x_t = torch.tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()

plt.xlabel("Hours")
plt.ylabel("Probability of Pass")
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.grid()
plt.show()
