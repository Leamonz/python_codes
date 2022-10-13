import numpy as np

n = 7
mat = np.zeros((n, n), dtype=np.int32)
num = 1
for time in range((n - 1) // 3 + 1):
    row = time
    col = 0
    while col < n - time * 2:
        mat[row][col] = num
        num += 1
        col += 1
    row += 1
    col -= 2
    while row < n - time * 2 and col >= time:
        mat[row][col] = num
        num += 1
        row += 1
        col -= 1
    row -= 2
    col += 1
    while row > time:
        mat[row][col] = num
        num += 1
        row -= 1

for i in range(n):
    for j in range(n):
        if mat[i][j] == 0:
            break
        print(format(mat[i][j], "-3d"), end=' ')
    print()
