import numpy as np

# 误差限
eps = 0.5e-5
e = 1e-8


# 方程1及其导数
def f1(x):
    return x ** 3 - x ** 2 - x - 1


def df1(x):
    return 3 * (x ** 2) - 2 * x - 1


# 方程2及其导数
def f2(x):
    return np.cos(x) - np.sin(x) - 0.5


def df2(x):
    return -np.sin(x) - np.cos(x)


def g(x):
    return x ** 2 - 7


def dg(x):
    return 2 * x


if __name__ == "__main__":
    # 求第一个方程的根
    # xk = 1.1
    # xk_1 = xk - f1(xk) / df1(xk)
    # while np.fabs(xk_1 - xk) > eps:
    #     xk = xk_1
    #     xk_1 = xk - f1(xk) / df1(xk)
    #     print(xk_1, end=' ')
    # print(f"\n方程x^3 - x^2 - x - 1 = 0 的正根为：{xk_1}")
    # print(f1(xk_1))

    # 求第二个方程的根
    # xk = 0
    # xk_1 = xk - f2(xk) / df2(xk)
    # while np.fabs(xk_1 - xk) > eps:
    #     xk = xk_1
    #     xk_1 = xk - f2(xk) / df2(xk)
    #     print(xk_1, end=' ')
    # print(f"\n方程cosx = 0.5 + sinx 的最小正根为：{xk_1}")
    # print(f2(xk_1))

    xk = 2.5
    xk_1 = xk - g(xk) / dg(xk)
    while np.fabs(xk_1 - xk) > eps:
        xk = xk_1
        xk_1 = xk - g(xk) / dg(xk)
        print(xk_1 - xk)
    print(xk)
    print(g(xk))
