# import torch
#
# # a = torch.arange(12).reshape(3, 4)
# # print(a)
# # arr = [0, 1, 3, 5, 4, 5, 8, 1, 7, 2, 8, 9]
# # b = torch.asarray(arr).reshape(3, 4)
# # print(b)
#
# # print(a == b)
# # print(a[a == b])
#
#
# a = torch.arange(8).reshape(4, 2)
# print(a)
# b = torch.arange(4).reshape(2, 2)
# print(b)
# print(a + b)

import cv2 as cv
import torch

# x = torch.arange(4.0, requires_grad=True)
# print(x)
# print(torch.dot(x, x).item())
# y = 2 * torch.dot(x, x)
# print(y)

# def f(a):
#     b = a * 2
#     while b.norm() < 1000:
#         b *= 2
#     if b.sum() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c
#
#
# a = torch.randn(size=(), requires_grad=True)
# print(a)
# d = f(a)
# print(d)
# d.backward()
# print(a.grad == d / a)
# print(a.grad)
# print(d / a)
# help(torch.ones)


# from torch.distributions import multinomial
#
# n = 10000
# fair_probs = torch.ones([6]) / 6
# print(fair_probs)
# counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# cum_counts = counts.cumsum(dim=0)
# estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# print(estimates)
# help(torch.cumsum)
import numpy as np
import torch

# help(np.linspace)
# help(torch.tensor.dtype)

# help(torch.nn.Linear)

# help(enumerate)
# help(torch.nn.Conv2d)
# a = np.array([1, 2, 3])
# a = np.append(a, 4)
# print(a)
if __name__ == "__main__":
    # a = np.arange(12).reshape(3, 4)
    # b = np.ones_like(a)
    # b1 = np.ones_like(a)
    # b2 = np.ones_like(a)
    # li = np.array([a, b, b1, b2])
    # print(li)
    # print(a)
    # print(b)
    # # c = np.vstack(li[:1] + li[2:])
    # print(li[:1])
    # print(li[2:])
    # c = li[:1] + li[2:]
    # print(c)
    pass

