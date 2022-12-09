import numpy as np
import matplotlib.pyplot as plt


def LagrangeInterpolate(xs, ys):
    def f(x):
        res = 0
        for j in range(len(ys)):
            sum = 1
            for i in range(len(xs)):
                if j == i:
                    continue
                else:
                    sum *= (x - xs[i]) / (xs[j] - xs[i])
            res += ys[j] * sum
        return res

    return lambda x: f(x)


if __name__ == "__main__":
    xs = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    ys = np.array([-0.916291, -0.693147, -0.510826, -0.357765, -0.223144])

    lagrange = LagrangeInterpolate(xs, ys)
    print(lagrange(0.54))
    print(np.log(0.54))
