import torch
import random
from torch.utils import data
from d2l import torch as d2l

# 1.获取数据
# 实际参数
true_w = torch.tensor([[2], [-3.4]])
true_b = 4.2


def synthetic_data(w, b, num_samples):
    x = torch.normal(0, 1, (num_samples, w.size(0)))
    y = torch.matmul(x, w) + b
    # 考虑了噪声（噪声符合正态分布N(0,0.01)）
    y += torch.normal(0, 0.01, y.shape)
    return x, y


x_data, y_data = synthetic_data(true_w, true_b, 1000)


# 初始化参数
# w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# b = torch.tensor([0.0], requires_grad=True)


# 读取数据集---定义一个数据生成器
def data_iter(batch_size, features, labels):
    num_samples = features.size(0)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        batch_indices = torch.tensor(indices[i:min(batch_size + i, num_samples)])
        yield features[batch_indices], labels[batch_indices]


# 2.建立模型
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


# def forward(x):
#     y_pred = torch.matmul(x, w) + b
#     return y_pred

model = LinearRegression()
# 3.构造损失函数和优化算法
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
# def loss(y_pred, y):
#     return ((y_pred - y.reshape(y_pred.shape)) ** 2) / 2


# 小批量随机梯度下降
# def sgd(params, lr, batch_size):
#     for param in params:
#         param = param - lr * param.grad / batch_size
#         param.retain_grad()


# 4.训练
for epoch in range(100):
    for x_val, y_val in data_iter(25, x_data, y_data):
        y_pred_val = model(x_val)
        loss = criterion(y_pred_val, y_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_l = criterion(model(x_val), y_val)
            print(f"epoch:{epoch + 1},loss:{format(train_l.mean(), '.5f')}")

print(f'w的估计误差: {(true_w - model.linear.weight.reshape(true_w.shape)).data}')
print(f'b的估计误差: {(true_b - model.linear.bias).data.item()}')
# lr = 0.01
# batch_size = 25
# for epoch in range(100):
#     for x_val, y_val in data_iter(10, x_data, y_data):
#         y_pred_val = forward(x_val)
#         l = loss(y_pred_val, y_val)
#         l.sum().backward()
#         sgd([w, b], lr=lr, batch_size=batch_size)
#         with torch.no_grad():
#             train_l = loss(forward(x_val), y_val)
#             print(f"epoch:{epoch + 1},loss:{format(train_l.mean(), '.5f')}")
#
# print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
# print(f'b的估计误差: {true_b - b}')
