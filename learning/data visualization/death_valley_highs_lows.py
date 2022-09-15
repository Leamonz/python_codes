import csv
from datetime import datetime

import matplotlib.pyplot as plt

filename = 'data/death_valley_2018_simple.csv'

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)

    for index, header in enumerate(header_row):
        print(index, header)
    highs, lows, dates = [], [], []
    for row in reader:
        current_date = datetime.strptime(row[2], '%Y-%m-%d')
        try:
            high = int(row[4])
            low = int(row[5])
        except ValueError:
            print(f"Missing data for {current_date}")
        else:
            highs.append(high)
            lows.append(low)
            dates.append(current_date)
plt.style.use('seaborn')
fig, ax = plt.subplots()

title = "2018年每日最高温度和最低温度\n美国加利福尼亚州死亡谷"
ax.set_title(title, fontsize=20)
ax.set_xlabel('', fontsize=16)
ax.set_ylabel('温度(F)', fontsize=16)
ax.plot(dates, highs, c='r')
ax.plot(dates, lows, c='b')
ax.fill_between(dates, highs, lows, color='b', alpha=.1)
fig.autofmt_xdate()
plt.ylim(20, 130)
ax.tick_params(axis='both', which='major', labelsize=16)
# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.show()
