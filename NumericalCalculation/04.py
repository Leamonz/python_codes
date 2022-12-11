import numpy as np


def f(x):
    return x ** 2 - 7


def autoderivation(f, interval=1e-5):
    """
    求f对x的导数值
    :param f: 求导函数
    :param x: 求导点
    :param interval:  选取间隔
    :return: x点的导数值（本质近似值）
    """
    return lambda x: (f(x + interval) - f(x - interval)) / (2 * interval)


def NewTonIterative(f, start, loss=1e-5 / 2):
    df = autoderivation(f, 1e-8)
    x_t = start
    x_t1 = x_t - (f(x_t) / df(x_t))
    print(f"""
            初始值：{start}，导数值：{df(x_t)} 迭代值：{x_t1}，误差：{np.fabs(x_t - x_t1)}，要求误差：{loss}
            """)
    while np.fabs(x_t - x_t1) > loss:
        x_t = x_t1
        x_t1 = x_t - (f(x_t) / df(x_t))
        print(f"""
            初始值：{start}，导数值：{df(x_t)} 迭代值：{x_t1}，误差：{np.fabs(x_t - x_t1)}，要求误差：{loss}
                """)
    return x_t1


x_pred = NewTonIterative(f, 2.5)
print(x_pred)
print(f(x_pred))
