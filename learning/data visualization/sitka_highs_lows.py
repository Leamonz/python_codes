import csv
from datetime import datetime

import matplotlib.pyplot as plt

filename = 'data/sitka_weather_2018_simple.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    for index, column_header in enumerate(header_row):
        print(index, column_header)
    # 收集每一行最高温度数据
    highs = []
    lows = []
    dates = []
    for row in reader:
        high = int(row[5])
        low = int(row[6])
        date = datetime.strptime(row[2], '%Y-%m-%d')
        highs.append(high)
        lows.append(low)
        dates.append(date)
plt.style.use('seaborn')
fig, ax = plt.subplots()
# 绘制最高温度和最低温度
ax.plot(dates, highs, c='r')
ax.plot(dates, lows, c='b')
# 填充两个y值系列直接的空隙
ax.fill_between(dates, highs, lows, facecolor='blue', alpha=.1)
# 设置y轴的范围
plt.ylim(20, 130)
# 让日期倾斜，不会相互重叠
fig.autofmt_xdate()
ax.set_title('2018年每日最高温度', fontsize=24)
ax.set_xlabel('', fontsize=16)
ax.set_ylabel('温度(F)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.show()
