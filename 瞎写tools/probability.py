import math

import matplotlib.pyplot as plt
import numpy as np


def C(m, n):
    return int(math.factorial(n) / math.factorial(m) / math.factorial(n - m))


class BinaryDistribution:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.probs = []
        for k in range(n + 1):
            prob = C(k, self.n) * (self.p ** k) * ((1 - self.p) ** (self.n - k))
            self.probs.append(prob)

    def __call__(self, k):
        return self.probs[k]

    def F(self, x):
        idx = 0
        F = 0
        while idx <= math.floor(x):
            F += self.probs[idx]
            idx += 1
        return F


class PoissonDistribution:
    def __init__(self, lamda):
        self.lamda = lamda

    def __call__(self, k):
        return np.power(self.lamda, k) / math.factorial(k) * np.exp(-self.lamda)


class HyperGeometryDistribution:
    def __init__(self, n, m, N):
        self.n = n
        self.m = m
        self.N = N
        self.probs = []
        for k in range(min(m, n) + 1):
            prob = C(k, m) / C(n, N) * C(n - k, N - m)
            self.probs.append(prob)

    def __call__(self, k):
        return self.probs[k]

    def F(self, x):
        idx = 0
        F = 0
        while idx <= int(x):
            F += self.probs[idx]
            idx += 1
        return F


class GeometryDistribution:
    def __init__(self, p):
        self.p = p

    def __call__(self, k):
        prob = self.p * ((1 - self.p) ** (k - 1))
        return prob

    def F(self, x):
        idx = 1
        F = 0
        while idx <= int(x):
            prob = self.p * ((1 - self.p) ** (idx - 1))
            F += prob
            idx += 1
        return F


if __name__ == "__main__":
    # problem = HyperGeometryDistribution(6, 4, 20)
    # print(problem.prob(0), problem.prob(1), problem.prob(2), problem.prob(3), problem.prob(4))

    # p = BinaryDistribution(15, 0.2)
    # print(p(3))
    # print(p(0))
    # print(1 - p(0) - p(1))
    # print(p(1) + p(2) + p(3))
    # print(p(0) + p(1))

    # bd = BinaryDistribution(5, 0.5)
    # print(bd.prob(1), bd.prob(0))
    # x = range(5)
    # F = []
    # for x in np.arange(5, step=0.001):
    #     F.append(bd.F(x))
    # plt.plot(F)
    # plt.show()

    # gd = GeometryDistribution(0.8)
    # print(gd.prob(3))

    p = np.exp(-0.4) / 1000
    bd = BinaryDistribution(6, p)
    print(bd(1))
    print(1 - bd(0))
