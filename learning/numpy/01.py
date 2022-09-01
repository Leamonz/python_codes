# arr = np.empty([4, 3], dtype=int)
# print(arr)
# arr = np.zeros([3, 2], dtype=int)
# arr = np.ones([2, 2], dtype=[("x", "i4"), ("y", "f4")])
# print(arr)
# arr = np.full(5, fill_value=6, dtype="i4")
# print(arr)
# arr = np.eye(10, dtype="i4")
# print(arr)
# arr = np.arange(0, 100, 5, dtype="i4").reshape([4, -1])
# print(arr)
# a = range(1, 6)
# it = iter(a)
# arr = np.fromiter(it, dtype=int)
# print(arr)
# arr = np.linspace(1, 10, 20, dtype="f4")
# print(arr)
# arr = np.logspace(1, 2, 10, dtype=float)
# print(arr)
# arr = np.random.randn(6)
# print(arr)

# li = range(1, 1000001)
# # li = []
# # for i in range(1000000):
# #     li.append(i)
#
# t1 = time.time()
# ret = sum(li)
# t2 = time.time()
# print(f"耗时{t2 - t1}")
#
# arr = np.array(li)
# t1 = time.time()
# ret = np.sum(arr)
# t2 = time.time()
# print(f"耗时{t2 - t1}")

# arr = np.array(range(0, 6))
# print(arr)
# print(arr.itemsize)
# print(arr.size)
# print(arr.shape)
# arr.shape = (3, -1)
# print(arr)
# print(arr.flags)
#
# arr1 = np.array(range(0, 11))
# arr = np.arange(10)
# print(arr)
# print(arr1)
# arr3 = arr1[2:7:2]
# arr2 = arr[2:7:2]
# print(arr2)
# print(arr3)

# arr = np.arange(15).reshape(3, 5)
# print(arr)
# print(arr[2, 1:])


# arr = np.arange(1, 13).reshape(-1, 3)
# print(arr)
# print(arr[[0, 1, 2], [0, 1, 0]])
# rows = np.array([[0, 0], [3, 3]])
# cols = np.array([[0, 2], [0, 2]])
# print(arr[rows, cols])

# arr1 = np.array([1, 2, 3, 4, 5])
# print(arr1[0, 2, 4]) 错误的

# 布尔索引
# arr1 = np.array([np.nan, 1, 2, np.nan, 4, 5, 6])
# print(arr1)
# print(arr1[~np.isnan(arr1)])

# 广播
# arr = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30]])
# print(arr)
# b = np.array([1, 2, 3])
# print(arr + b)
# bb = np.tile(b, (4, 1))
# print(bb)
# print(arr + bb)

# arr = np.arange(12).reshape(4, 3)
# arr1 = arr.T
# print(arr1)
# print(arr)
# print("\n")
# for x in np.nditer(arr):
#     print(x, end=", ")

# a = np.arange(12).reshape(4, 3)
# print(a, '\n')
# c = a.flatten()  # 相当于拷贝，通过c修改数组中的元素不会影响原数组
# print(c)
# c[0] = 100
# print(c)
# print(a)
# b = a.ravel()  # 相当于引用，通过b修改数组中元素的值也会改变原数组相应元素的值
# print(b)
# b[0] = 100
# print(b)
# print(a)
# for x in a.flat:
#     print(x, end=", ")


import numpy as np

# a = np.arange(12).reshape(3, 4)
# print(a, '\n')
# b = np.transpose(a)
# print(b, '\n')
# x = np.arange(1, 13).reshape(1, 3, 1, 4)
# print(x)
# print(x.shape, x.ndim, '\n')
# y = np.squeeze(x)  # 只删除值为1的轴
# print(y)
# print(y.shape, y.ndim, '\n')

x = np.arange(1, 5).reshape(2, 2)
y = np.arange(5, 9).reshape(2, 2)
print(x, '\n')
print(y, '\n')
output = np.concatenate((x, y))
print(output, '\n')
output1 = np.concatenate((x, y), axis=1)
print(output1, '\n')
