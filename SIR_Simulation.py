import matplotlib.pyplot as plt
import numpy as np

N = 1e7
# 患病者的有效接触率
lamda = 0.8
# 迭代时间
T = 70
# 易感者比例
s = np.zeros(T + 1)
# 感染者比例
i = np.zeros(T + 1)
# 假设初始患病人数为45
i[0] = 45.0 / N
s[0] = 1 - i[0]

for t in range(T):
    i[t + 1] = i[t] + i[t] * s[t] * lamda
    s[t + 1] = 1 - i[t + 1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(i, c='r', lw=2)
ax.plot(s, c='b', lw=2)

ax.set_xlabel("Day", fontsize=20)
ax.set_ylabel("Infective Ratio", fontsize=20)
ax.grid(1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
