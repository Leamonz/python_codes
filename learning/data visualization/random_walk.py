from random import choice


# 随机漫步
class RandomWalk:
    def __init__(self, num_points=5000):
        self.num_points = num_points
        self.x = [0]
        self.y = [0]

    def fill_walk(self):
        while len(self.x) < self.num_points:
            # 决定移动方向以及移动距离
            x_direction = choice([1, -1])
            x_distance = choice(range(0, 5))
            x_step = x_distance * x_direction

            y_direction = choice([1, -1])
            y_distance = (choice(range(0, 5)))
            y_step = y_distance * y_direction
            # 拒绝原地踏步
            if x_step == 0 and y_step == 0:
                continue
            # 计算下一个点的坐标
            next_x = self.x[-1] + x_step
            next_y = self.y[-1] + y_step
            self.x.append(next_x)
            self.y.append(next_y)
