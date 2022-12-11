import numpy as np

A = np.array([[0.8147, 0.0975, 0.1576, 0.1419, 0.6557],
              [0.9058, 0.2785, 0.9706, 0.4218, 0.0357],
              [0.1270 * 1e10, 0.5469, 0.9572, 0.9157, 0.8491],
              [0.9134, 0.9575, 0.4854 * 1e8, 0.7922, 0.9340],
              [0.6324, 0.9649, 0.8003, 0.9595, 0.6787]])

b = np.array([0.000000002258000, 0.000000001597700, 1.270000002354900, 0.024270003904200, 0.000000003360250])
b = 1e9 * b
b = b.reshape(-1, 1)


def Max(mat, row):
    rows, cols = mat.shape[0], mat.shape[1]
    max = mat[row][row]
    idx = row
    for r in range(row + 1, rows):
        if mat[r][row] > max:
            idx = r
            max = mat[r][row]
    return idx


def Swap_Row(mat, ori_Row, tar_Row):
    # ori_Row = pos[0]
    # tar_Row = pos[1]
    cols = mat.shape[1]
    for col in range(cols):
        mat[ori_Row][col], mat[tar_Row][col] = mat[tar_Row][col], mat[ori_Row][col]
    return mat


def Gauss(mat):
    rows, cols = mat.shape[0], mat.shape[1]
    for r in range(rows):
        max_idx = Max(mat, r)
        if max_idx != r:
            mat = Swap_Row(mat, r, max_idx)
        for rr in range(r + 1, rows):
            m = mat[rr][r] / mat[r][r]
            mat[rr] -= m * mat[r]
    ans = np.zeros(rows)
    for i in reversed(range(ans.shape[0])):
        sum = mat[i][-1]
        for j in range(i, rows):
            if j > i:
                sum -= mat[i][j] * ans[j]
        ans[i] = sum / mat[i][i]
    return ans


if __name__ == "__main__":
    B = np.concatenate((A, b), axis=1)
    target = np.linalg.inv(A) @ b
    target = target.flatten()
    print("准确值:")
    print(target)
    x_pred = Gauss(B)
    print("经过高斯消元所得值:")
    print(x_pred)
