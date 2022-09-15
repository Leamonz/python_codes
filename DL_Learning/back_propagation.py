import random

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# w = torch.tensor([1.0])
# w.requires_grad = True  # 表示需要计算梯度
# alpha = 0.05  # 学习率


# def forward(x):
#     return w * x
#
#
# def loss(x, y):
#     y_pred = forward(x)
#     return (y_pred - y) ** 2
#
#
# print("Prediction(before training):", 4, forward(4).item())
# for epoch in range(100):
#     l = 0
#     for x_val, y_val in zip(x_data, y_data):
#         l = loss(x_val, y_val)
#         l.backward()  # 生成计算图 进行反馈计算
#         # 计算的梯度保存在w中---w.grad
#         print(f"{x_val}, {y_val}, loss:{l.data.item()}, grad:{w.grad.item()}")
#         w.data = w.data - alpha * w.grad.data
#         # 梯度计算完之后不会自动清零，而是会累加，因此要手动清零
#         w.grad.data.zero_()
#     print("progress:", epoch, l.item())
#
# print("Prediction(after training):", 4, forward(4).item())

# y = w1 * (x ** 2) + w2 * x + b
w1 = torch.tensor([1.0])
w2 = torch.tensor([2.0])
b = torch.tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

alpha = 0.01


def forward(x):
    return w1 * (x ** 2) + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("Prediction(before training):", 4, forward(4).item())
for epoch in range(100):
    l = 0
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print(f"x:{x_val}\ty:{y_val}\tgrad-w1:{w1.grad.item()}\tgrad-w2:{w2.grad.item()}\tgrad-b:{b.grad.item()}")
        w1.data = w1.data - alpha * w1.grad.data
        w2.data = w2.data - alpha * w2.grad.data
        b.data = b.data - alpha * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("\n\t\tProgress:", epoch, l.item(), "\n")

print("Prediction(after training):", 4, forward(4).item())
