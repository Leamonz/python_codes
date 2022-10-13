import numpy as np

# A = np.array([[1, 2, 3], [2, 1, 2], [1, 3, 4]])
# print(A)
# print(np.linalg.inv(A))

# 误差
eps = 1e-8


def LDU_Decompose(mat):
    L = np.zeros_like(mat)
    D = np.zeros_like(mat)
    U = np.zeros_like(mat)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i > j:
                L[i][j] = mat[i][j]
            elif i < j:
                U[i][j] = mat[i][j]
            else:
                D[i][j] = mat[i][j]
    return L, D, U


# 几种迭代方法的矩阵表示法
def iter_Jacobi(x0, L, D, U, b):
    xk = x0
    xk_1 = -np.matmul(np.matmul(np.linalg.inv(D), (L + U)), xk) + np.matmul(np.linalg.inv(D), b)
    while np.linalg.norm(xk_1 - xk) > eps:
        xk = xk_1
        xk_1 = -np.matmul(np.matmul(np.linalg.inv(D), (L + U)), xk) + np.matmul(np.linalg.inv(D), b)
    return xk_1


def iter_Gauss_Siedel(x0, L, D, U, b):
    xk = x0
    xk_1 = -np.matmul(np.matmul(np.linalg.inv(D + L), U), xk) + np.matmul(np.linalg.inv(D + L), b)
    while np.linalg.norm(xk_1 - xk) > eps:
        xk = xk_1
        xk_1 = -np.matmul(np.matmul(np.linalg.inv(D + L), U), xk) + np.matmul(np.linalg.inv(D + L), b)
    return xk_1


if __name__ == "__main__":
    A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
    L, D, U = LDU_Decompose(A)
    b = np.array([24, 30, -24])
    x0 = [0, 0, 0]
    x_pred1 = iter_Jacobi(x0, L, D, U, b)
    print(x_pred1)
    print(np.matmul(A, x_pred1))
    x_pred2 = iter_Gauss_Siedel(x0, L, D, U, b)
    print(x_pred2)
    print(np.matmul(A, x_pred2))
