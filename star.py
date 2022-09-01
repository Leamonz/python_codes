import sys
from random import randint

import pygame
from pygame.sprite import Sprite


class Starry_Sky:
    def __init__(self):
        self.screen = pygame.display.set_mode((1200, 750))
        pygame.display.set_caption("Starry Sky")
        self.stars = pygame.sprite.Group()
        self.number_stars = 100
        self.create_stars()
        self.bg_color = (230, 230, 230)

    def check_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

    def create_stars(self):
        for number in range(self.number_stars):
            new_star = Star(self)
            new_star.x = randint(0, 1200)
            new_star.y = randint(0, 750)
            new_star.rect.x = new_star.x
            new_star.rect.y = new_star.y
            self.stars.add(new_star)

    def update_event(self):
        self.screen.fill(self.bg_color)
        # self.create_stars()
        self.stars.draw(self.screen)
        pygame.display.update()

    def run(self):
        while True:
            self.check_event()
            for star in self.stars.sprites():
                star.update()
            pygame.time.delay(800)
            self.update_event()


class Star(Sprite):
    def __init__(self, starry_sky):
        super().__init__()
        self.screen = starry_sky.screen
        self.image = pygame.image.load('star-fill.png')
        self.rect = self.image.get_rect()
        self.rect.x = self.rect.width
        self.rect.y = self.rect.height

        self.x = float(self.rect.x)
        self.y = float(self.rect.y)

    def update(self):
        self.x = randint(0, 1200)
        self.y = randint(0, 750)
        self.rect.x = self.x
        self.rect.y = self.y


ss = Starry_Sky()
ss.run()
