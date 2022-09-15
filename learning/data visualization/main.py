# import matplotlib.pyplot as plt
#
# from random_walk import RandomWalk
#
# # 绘制折线图
# # input_values = [1, 2, 3, 4, 5]
# # squares = [x ** 2 for x in input_values]
# # plt.style.use('seaborn')
# # fig, ax = plt.subplots()
# # ax.plot(input_values, squares, linewidth=3)
# # # 设置图标标题并给坐标轴加上标签
# # ax.set_title("平方值", fontsize=24)
# # ax.set_xlabel("值", fontsize=14)
# # ax.set_ylabel("值的平方", fontsize=14)
# # # 设置刻度标记的大小
# # ax.tick_params(axis='both', labelsize=14)
# # 利用scatter()绘制散点图
# # plt.style.use('Solarize_Light2')
# # fig, ax = plt.subplots()
# # # x = [1, 2, 3, 4, 5]
# # # y = [1, 4, 9, 16, 25]
# #
# # x = range(1, 1001)
# # y = [i ** 3 for i in x]
# # # s可以设置绘制的点的尺寸, c可以设置绘制的点的颜色
# # # ax.scatter(x, y, s=5, color=(0, 0.8, 0))
# # ax.scatter(x, y, c=y, cmap=plt.cm.YlOrBr, s=10)
# # # 设置图标标题并给坐标轴加上标签
# # ax.set_title("立方值", fontsize=24)
# # ax.set_xlabel("值", fontsize=14)
# # ax.set_ylabel("值的立方", fontsize=14)
# #
# # # 设置刻度标记的大小
# # ax.tick_params(axis='both', labelsize=14)
# # # 设置坐标轴的取值范围
# # ax.axis([0, 1100, 0, 1e9])
#
# rw = RandomWalk()
# rw.fill_walk()
# plt.style.use('classic')
# fig, ax = plt.subplots()
# point_numbers = range(rw.num_points)
# ax.scatter(rw.x, rw.y, c=point_numbers, cmap=plt.cm.autumn, edgecolor='none',
#            s=10)
# # 隐藏坐标轴
# # ax.get_xaxis().set_visible(False)
# # ax.get_yaxis().set_visible(False)
# # 汉字字体，优先使用楷体，找不到则使用黑体
# plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
#
# # 正常显示负号
# plt.rcParams['axes.unicode_minus'] = False
# plt.show()

# from plotly import offline
# from plotly.graph_objs import Bar, Layout
#
# from dice import Dice
#
# dice1 = Dice()
# dice2 = Dice()
# num = []
# for i in range(5000):
#     num.append(dice1.roll() + dice2.roll())
# frequencies = []
# for value in range(2, dice1.num_sides + dice2.num_sides + 1):
#     frequency = num.count(value)
#     frequencies.append(frequency)
# # 利用直方图对结果进行可视化
# x_values = list(range(2, dice1.num_sides + dice2.num_sides + 1))
# data = [Bar(x=x_values, y=frequencies)]
#
# x_axis_config = {'title': '结果', 'dtick': 1}
# y_axis_config = {'title': '结果的频率'}
# my_layout = Layout(title='掷两个D6 5000次的结果', xaxis=x_axis_config, yaxis=y_axis_config)
# offline.plot({'data': data, 'layout': my_layout}, filename='两个d6.html')
