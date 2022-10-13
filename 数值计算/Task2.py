import numpy as np

A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
b = np.array([24, 30, -24])


def iter_Jacobi(x):
    xk = x
    xk_1 = np.zeros(len(x))
    for i in range(16):
        print(f"{i + 1}th iteration: {xk}")
        for k in range(len(xk)):
            xk_1[k] = b[k]
            for j in range(len(xk)):
                if j == k:
                    continue
                xk_1[k] -= A[k][j] * xk[j]
            xk_1[k] /= A[k][k]
        xk = xk_1
    return xk_1


def iter_Gauss_Siedel(x):
    xk = x
    xk_1 = np.zeros(len(x))
    for i in range(16):
        print(f"{i + 1}th iteration: {xk}")
        for k in range(len(xk)):
            xk_1[k] = b[k]
            for j in range(len(xk)):
                if j == k:
                    continue
                if j < k:
                    xk_1[k] -= A[k][j] * xk_1[j]
                else:
                    xk_1[k] -= A[k][j] * xk[j]
            xk_1[k] /= A[k][k]
        xk = xk_1
    return xk_1


def iter_SOR(x, w=1.0):
    xk = x
    xk_1 = np.zeros(len(x))
    for i in range(16):
        print(f"{i + 1}th iteration: {xk}")
        for k in range(len(xk)):
            xk_1[k] = (1 - w) * xk[k] + b[k] / A[k][k] * w
            for j in range(len(xk)):
                if j == k:
                    continue
                if j < k:
                    xk_1[k] -= (A[k][j] * xk_1[j]) / A[k][k] * w
                else:
                    xk_1[k] -= (A[k][j] * xk[j]) / A[k][k] * w
        xk = xk_1
    return xk_1


if __name__ == "__main__":
    x0 = np.array([1, 2, -1])
    target = np.linalg.inv(A) @ b.reshape(-1, 1)
    print(f'target: {target.flatten()}')
    x_pred1 = iter_Jacobi(x0)
    print()
    print(x_pred1)
    x_pred2 = iter_Gauss_Siedel(x0)
    print()
    print(x_pred2)
    x_pred3 = iter_SOR(x0, w=1.8)
    print()
    print(x_pred3)
    x_pred4 = iter_SOR(x0, w=1.22)
    print()
    print(x_pred4)
