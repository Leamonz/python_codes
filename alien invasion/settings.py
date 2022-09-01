class Settings:
    def __init__(self):
        self.width = 1200
        self.height = 750
        self.color = (230, 230, 230)
        # 子弹（小矩形）的属性
        self.bullet_width = 4
        self.bullet_height = 15
        self.bullet_color = (60, 60, 60)

        # 外星人下落速度
        self.fleet_drop_speed = 5
        # 外星人移动方向，1表示右移，-1表示左移；右移要增加x坐标，而左移要减小x坐标
        # 加快游戏节奏的速度
        self.speedup_scale = 1.2
        # 外星人分数提高的速度
        self.score_scale = 20
        # 子弹变化速度
        self.bullet_scale = 4
        self.init_dynamic_settings()

    def init_dynamic_settings(self):
        # 初始化会随游戏进行而变化的量
        self.grade = 1
        self.ship_speed = 0.7
        self.bullet_speed = 1
        self.alien_speed = 0.2
        self.fleet_direction = 1
        # 随着游戏的进行，外星人的分数会变高
        self.alien_points = 20
        self.bullet_number = 1

    def increase_speed(self):
        self.bullet_speed *= self.speedup_scale
        self.alien_speed *= self.speedup_scale
        if self.grade < 10:
            if self.grade < 5:
                self.ship_speed *= self.speedup_scale
            self.bullet_width += self.bullet_scale
            self.alien_points += self.score_scale
            self.grade += 1
        if self.bullet_number < 5:
            self.bullet_number += 1
        print(self.bullet_number)
