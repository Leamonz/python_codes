import pygame.font
from pygame.sprite import Group

from ship import Ship


class ScoreBoard:
    def __init__(self, game):
        self.game = game
        self.screen = game.screen
        self.screen_rect = self.screen.get_rect()
        self.settings = game.settings
        self.stats = game.stats

        self.text_color = (30, 30, 30)
        self.font = pygame.font.SysFont(None, 48)
        self.prep_Score()
        self.prep_HighestScore()
        self.prep_grade()
        self.prep_ships()

    # 渲染
    def prep_Score(self):
        score_str = "{:,}".format(self.stats.score)
        self.score_image = self.font.render(score_str, True, self.text_color, self.settings.color)
        # 在屏幕右上方显示分数
        self.score_image_rect = self.score_image.get_rect()
        self.score_image_rect.right = self.screen_rect.right - 20
        self.score_image_rect.top = 20

    def prep_HighestScore(self):
        highest_score_str = "{:,}".format(self.stats.highest_score)
        self.highest_score_image = self.font.render(highest_score_str, True, self.text_color, self.settings.color)
        # 将最高分数放在屏幕顶部中央
        self.highest_score_image_rect = self.highest_score_image.get_rect()
        self.highest_score_image_rect.top = self.screen_rect.top + 20
        self.highest_score_image_rect.centerx = self.screen_rect.centerx

    # 显示文本
    def show_Score(self):
        self.screen.blit(self.score_image, self.score_image_rect)
        self.screen.blit(self.highest_score_image, self.highest_score_image_rect)
        self.screen.blit(self.grade_image, self.grade_rect)
        self.ships.draw(self.screen)

    # 检查是否产生了新的最高分
    def check_highest_score(self):
        if self.stats.score > self.stats.highest_score:
            self.stats.highest_score = self.stats.score
            self.prep_HighestScore()

    # 显示玩家当前等级
    def prep_grade(self):
        grade_str = str(self.settings.grade)
        self.grade_image = self.font.render(grade_str, True, self.text_color, self.settings.color)
        # 将等级放在得分的下方
        self.grade_rect = self.grade_image.get_rect()
        self.grade_rect.right = self.score_image_rect.right
        self.grade_rect.top = self.score_image_rect.bottom + 10

    # 显示剩余飞船数量
    def prep_ships(self):
        self.ships = Group()
        for ship_number in range(self.stats.ships_left):
            ship = Ship(self.game)
            ship.rect.x = 10 + ship_number * ship.rect.width
            ship.rect.y = 10
            self.ships.add(ship)
