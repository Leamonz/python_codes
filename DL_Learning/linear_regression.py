# import torch

# A = torch.arange(20).reshape(5, -1)
# print(A)
#
# sum_A = A.sum(axis=1, keepdims=True)
# print(A / sum_A)
#
# print(A.cumsum(axis=0))
# x = torch.arange(24).reshape(2, 3, 4)
# print(x)
# # print(len(x))
# print(x.sum(axis=0))
# print(x.sum(axis=1))
# print(x.sum(axis=2))

import torch
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 传的两个参数分别为权重(w)和偏置(b)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
# 定义损失函数----参数表明Loss求和降维成标量，求和后不用算均值
criterion = torch.nn.MSELoss(reduction='sum')
# lr=learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training Cycle
for epoch in range(1000):
    y_pred_data = model(x_data)
    loss = criterion(y_pred_data, y_data)
    print("Progress:", epoch, loss.item())

    optimizer.zero_grad()  # 手动梯度清零
    loss.backward()  # 反馈操作
    optimizer.step()  # 对参数进行更新

x_test = torch.tensor([4.0])
y_test = model(x_test)
print("Prediction(after training):", 4, y_test.data.item())
print(model.linear.weight.item(), model.linear.bias.item())
