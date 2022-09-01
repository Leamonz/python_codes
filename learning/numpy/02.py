import numpy as np

# x = np.arange(12)
# print(x)
# a = np.split(x, [2, 7])
# print(a)
# a = np.arange(4).reshape(2, 2)
# b = np.array(range(4, 8)).reshape(2, 2)
# print(a, '\n')
# print(b, '\n')
# c = np.stack((a, b))
# print(c, c.ndim)
# x = np.arange(6).reshape(2, -1)
# print(x, '\n')
# y = np.append(x, [7, 8, 9])
# print(y)
# y = np.append(x, [[7, 8, 9], [10, 11, 12]], axis=1)
# print(y)
# y = np.resize(x, (3, 2))  可以广播
# y = x.reshape(3, 3)
# print(y)

# x = np.arange(6).reshape(3, 2)
# print(x, '\n')
# a = np.insert(x, 2, [6, 7, 8], axis=1)
# print(a)

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 11, 2, 33, 3, 4])
# print(x)
# a = np.unique(x, return_counts=True)[0]
# print(a)
# b = np.unique(x, return_counts=True)[1]
# print(b)

# a = np.array([0, 30, 60, 90, 180, 360])
# sin = a * np.pi / 180
# print(sin)
# print(np.sin(sin))
# print(np.arcsin(np.sin(sin)))
# print(np.degrees(np.arcsin(np.sin(sin))))

# a = np.arange(12).reshape(3, 4)
# b = np.arange(12, 24).reshape(3, 4)
# print(a)
# print(b)
# c = a * b
# print(c)
# np.random.randint
# x = np.array([[10, 7, 4, 5], [3, 2, 1, 8]])
# print(x)
# print(np.mean(x))
# print(np.mean(x, axis=0))
# print(np.mean(x, axis=1))
# print(x.mean(axis=0))
# print(x.mean(axis=1))
# print(np.percentile(x, 50))
# print(np.percentile(x, 50, axis=1))
# print(np.percentile(x, 50, axis=0))

# y = np.random.randint(1, 11, 9)
# print(y)
# print(sorted(y))
# print(np.percentile(y, 50))
# print(np.median(y))
#
# x = np.array([1, 2, 3, 4])
# wts = np.array([4, 3, 2, 1])
# print(x)
# print(np.average(x))
# # 加权平均和
# print(np.average(x, weights=wts))

# y = np.array([46, 57, 23, 39, 1, 10, 0, 120])
# print(y, '\n')
# a = np.argpartition(y, 2)
# print(y[a])
# b = np.argpartition(y, 3)
# print(y[b])
# c = np.argpartition(y, -2)
# print(y[c])
x = np.arange(1, 7)
print(x)
print(id(x))
# y = x
# print(y)
# print(id(y))
# y.shape = (3, 2)
# print(y)
# print(x)
y = x.view()
print(y)
print(id(y))
y[2] = 30
print(x, '\n', y)
z = x[2:]
print(z)
z[1] = 400
print(z)
print(y)
print(x)

a = slice(x, 2)
print(a)
