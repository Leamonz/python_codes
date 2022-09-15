import random

import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# def forward(x, w):
#     return x * w
#
#
# def loss(x, y, w):
#     y_pred = forward(x, w)
#     return (y_pred - y) ** 2
#
#
# w_list = []
# mse_list = []
#
# for w_val in np.arange(0.0, 4.0, 0.1):
#     l_sum = 0
#     for x_val, y_val in zip(x_data, y_data):
#         y_pred_val = forward(x_val, w_val)
#         l_sum += loss(x_val, y_val, w_val)
#     w_list.append(w_val)
#     mse_list.append(l_sum / 3)
#     print("w: ", format(w_val, ".4f"))
#     print("MSE: ", format(l_sum / 3, ".4f"))
#     print()
#
# fig, ax = plt.subplots()
# plt.plot(w_list, mse_list)
# plt.xlabel("w")
# plt.ylabel("MSE")
# plt.show()

# 梯度下降算法
# 给w随机赋一个初始值
w = 1.0
# 学习率(尽量小)---超参数
alpha = 0.01


def forward(x):
    return x * w


# 随机梯度下降---随机取一个值而非用所有数据点
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


w_list = []
epochs = range(1, 101)
for epoch in epochs:
    idx = random.randint(0, 2)
    x_val = x_data[idx]
    y_val = y_data[idx]
    w -= alpha * gradient(x_val, y_val)
    w_list.append(w)
    l = loss(x_val, y_val)
    print(f"Epochs: {epoch} w:{w} loss:{l}")

# def cost(lx, ly):
#     cost = 0
#     for x_val, y_val in zip(lx, ly):
#         y_pred = forward(x_val)
#         cost += (y_pred - y_val) ** 2
#     return cost / len(lx)
#
#
# # 求梯度
# def gradient(lx, ly):
#     grad = 0
#     for x_val, y_val in zip(lx, ly):
#         grad += 2 * x_val * (x_val * w - y_val)
#     return grad / len(lx)
#
#
# w_list = []
# print("Predict(before training):", 4, forward(4))
# epochs = range(1, 101)
# for epoch in epochs:  # 训练100epoch
#     cost_val = cost(x_data, y_data)
#     w = w - alpha * gradient(x_data, y_data)
#     w_list.append(w)
#     print("Epoch:", epoch, " w:", w, " loss:", cost_val)
# print("Predict(after training):", 4, forward(4))

fig, ax = plt.subplots()
ax.plot(epochs, w_list)
plt.xlabel("Epochs")
plt.ylabel("w")
plt.show()
