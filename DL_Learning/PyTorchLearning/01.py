import torch
import numpy as np

# x = np.arange(12).reshape(3, -1)
# y = np.arange(12, 24).reshape(3, -1)
# x = torch.from_numpy(x)
# y = torch.from_numpy(y)
# print(x)
# print(y)
# z = torch.cat((x, y), dim=1)
# print(z)
# print(z.shape)
# p = torch.stack((x, y), dim=1)
# print(p)
# print(p.shape)

tensor = torch.ones(4, 4)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# @表示矩阵乘法
y1 = tensor @ tensor.T
# tensor调用matmul方法与tensor.T进行矩阵乘法
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
print(y2)
print(y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
# tensor调用mul方法与tensor进行元素对应相乘
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z2)
print(z3)
