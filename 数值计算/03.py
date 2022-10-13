import math

import numpy as np
import matplotlib.pyplot as plt

# 迭代方法


eps = 1e-10


def f(x):
    return x ** 3 - 0.2 * (x ** 2) - 0.2 * x - 1.2


# 弦截法
# 单点弦法
def SinglePointIter(x1, x0):
    xk = x1
    xk_1 = xk - f(xk) / (f(xk) - f(x0)) * (xk - x0)
    while math.fabs(xk_1 - xk) > eps:
        xk = xk_1
        xk_1 = xk_1 = xk - f(xk) / (f(xk) - f(x0)) * (xk - x0)
    return xk_1


# 双点弦法---速度快于单点弦法，稍慢于牛顿迭代法
def DualPointIter(x1, x0):
    xk = x1
    xk_1 = x0
    while math.fabs(xk_1 - xk) > eps:
        xk_0 = xk
        xk = xk_1
        xk_1 = xk - f(xk) / (f(xk) - f(xk_0)) * (xk - xk_0)
    return xk_1


if __name__ == "__main__":
    x_pred = SinglePointIter(1, 1.5)
    print(x_pred)
    print(f(x_pred))
    x_pred_1 = DualPointIter(1, 1.5)
    print(x_pred_1)
    print(f(x_pred_1))
