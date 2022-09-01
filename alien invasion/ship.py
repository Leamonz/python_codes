import pygame
from pygame.sprite import Sprite

class Ship(Sprite):
    def __init__(self, game):
        super().__init__()
        # 初始化飞船
        self.screen = game.screen
        self.screen_rect = game.screen.get_rect()
        # 导入飞船图片并获取其外接矩形
        self.image = pygame.image.load("images/ship.bmp")
        self.rect = self.image.get_rect()
        # 将飞船放在屏幕底部中央（底部中央对齐）
        self.rect.midbottom = self.screen_rect.midbottom
        self.settings = game.settings
        self.x = float(self.rect.x)
        self.moving_right = False
        self.moving_left = False

    def update(self):
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.x += self.settings.ship_speed
        if self.moving_left and self.rect.left > 0:
            self.x -= self.settings.ship_speed
        self.rect.x = self.x

    def blitme(self):
        self.screen.blit(self.image, self.rect)

    # 将飞船居中显示
    def center_ship(self):
        self.rect.midbottom = self.screen_rect.midbottom
        self.x = float(self.rect.x)
