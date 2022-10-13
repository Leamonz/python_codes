import numpy as np
import matplotlib.pyplot as plt


def metric(accuracy, runtime, memory):
    return (accuracy - 0.5 * runtime - 0.5 * memory) / 100


output = np.zeros(4)
accuracy = np.array([97, 99, 97, 98])
runtime = np.array([1, 13, 3, 9])
memory = np.array([3, 9, 2, 9])
output = metric(accuracy, runtime, memory)

plt.plot(output)
plt.xlabel("order")
plt.ylabel("metric")
plt.show()
