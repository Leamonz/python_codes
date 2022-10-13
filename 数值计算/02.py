from matplotlib import pyplot as plt
import numpy as np


# 辅助判断全局收敛性
# def f(x):
#     return (2 / 3) * ((2 * x + 5) ** (-2 / 3))
#
#
# x = np.arange(2, 3, step=0.1)
# y = f(x)
#
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y_value')
# plt.show()
# print(f(2))


def f(x):
    return x ** 3 - 10


def df(x):
    return 3 * (x ** 2)


def g(x):
    return x - f(x) / df(x)


list = []


def NewtonIter(x):
    xk = x
    xk_1 = g(xk)
    list.append(xk_1)
    for i in range(5):
        xk = xk_1
        xk_1 = g(xk)
        list.append(xk_1)
    return xk_1


ans = NewtonIter(2)
print(ans)
print(ans ** 3)
print(f(ans))

times = [i + 1 for i in range(len(list))]
plt.plot(times, list)
plt.xlabel("iteration")
plt.ylabel("x_value")
plt.show()
