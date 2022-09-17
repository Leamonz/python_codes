from matplotlib import pyplot as plt
import matplotlib
import numpy as np


def f(x):
    return x ** 2 - x - 1


# a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
#
# for i in a:
#     print(i)

plt.rcParams['font.family'] = ['STFangsong']

font1 = {
    'color': 'blue',
    'size': 18
}
xpoints = np.arange(0, 5, step=0.1)
ypoints = f(xpoints)
# plt.plot(xpoints, ypoints, 'r', marker='o', ms=20, mfc='c', mec='m')

# plt.plot(xpoints, ypoints, linestyle='dashed')
# matplotlib默认不显示中文，如果要显示中文需要外部下载的字体文件
# plt.xlabel("x轴")
# plt.ylabel("y轴")
# fontdict可以通过自定义字典设置字体样式
# plt.title("标题", fontdict=font1)
# plt.grid(which='minor',axis=)
# 此处的fig和ax分别是figure类和axes类的实例化
# 实例化两个子图，（1,2）表示一行两列
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(xpoints, ypoints, label='trend')
# ax[0].set_title('title 1')
# ax[1].plot(xpoints, ypoints, color='cyan')
# ax[1].set_title('title 2')

# plt.figure(1)
# 12表示子图分布：一行两列；最后一个1表示第一个子图
# plt.subplot(121)
# plt.plot(xpoints, ypoints, label='trend')
# plt.title('title 1', fontsize=12, c='r')
#
# plt.subplot(122)
# plt.plot(xpoints, ypoints, c='cyan')
# plt.title('title 2', fontsize=12, c='r')

# fig, ax = plt.subplots(1, 3, sharex='row', sharey='row')
# ax[0].set_xlabel('x - label', fontsize=20)
# ax[0].set_ylabel('y - label', fontsize=20)
# ax[0].plot(xpoints, ypoints, c='cyan')
# ax[0].set_title('pic1', fontsize=18)
# ax[0].grid(which='major', axis='both')
#
# ax[1].plot(xpoints, ypoints, c='green')
# ax[1].set_title('pic2', fontsize=18)
# ax[1].grid(axis='x')
#
# ax[2].plot(xpoints, ypoints, c='r')
# ax[2].set_title('pic3')
# ax[2].grid(which='both', axis='y')

# x = np.arange(0, 3, step=0.1)
# y = [i ** 2 for i in x]
# # 第一幅子图
# plt.subplot(1, 2, 1)
# plt.plot(x, y, c='cyan')
# plt.title('pic1')
# # 第二幅子图
# plt.subplot(1, 2, 2)
# plt.plot(x, y, lw=5)
# plt.title('pic2')
#
# plt.suptitle('subplot test', fontsize=20)

# 创建一些测试数据
# x = np.linspace(0, 2 * np.pi, 400)
# y = np.sin(x ** 2)
#
# # 创建一个画像和子图 -- 图1
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title('Simple plot')
#
# # 创建两个子图 -- 图2
# f, (ax1, ax2) = plt.subplots(1, 2, sharey='all')
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)
#
# # 创建四个子图 -- 图3
# fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
# axs[0, 0].plot(x, y)
# axs[1, 1].scatter(x, y)
#
# # 共享 x 轴
# plt.subplots(2, 2, sharex='col')
#
# # 共享 y 轴
# plt.subplots(2, 2, sharey='row')
#
# # 共享 x 轴和 y 轴
# plt.subplots(2, 2, sharex='all', sharey='all')
#
# # 这个也是共享 x 轴和 y 轴
# plt.subplots(2, 2, sharex='all', sharey='all')
#
# # 创建标识为 10 的图，已经存在的则删除
# fig, ax = plt.subplots(num=10, clear=True)

x = np.arange(0, 5, step=0.1)
y = [i ** 2 for i in x]
colors = np.arange(0, 100, step=2)
plt.scatter(x, y, c=colors, vmin=10, vmax=200)
plt.colorbar()

plt.show()
