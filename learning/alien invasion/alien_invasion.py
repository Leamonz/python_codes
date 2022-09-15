import sys
from time import sleep

import pygame

from alien import Alien
from bullet import Bullet
from button import Button
from game_stats import GameStats
from scoreboard import ScoreBoard
from settings import Settings
from ship import Ship


class AlienInvasion:
    def __init__(self):
        pygame.init()

        self.settings = Settings()
        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height))

        pygame.display.set_caption("Alien Invasion")

        self.ship = Ship(self)
        # 用一个编组来存储子弹，编组类似于列表
        self.bullets = pygame.sprite.Group()
        # 用一个编组来存储外星人
        self.aliens = pygame.sprite.Group()
        self.create_fleet()
        # 记录游戏状态
        self.stats = GameStats(self)
        self.play_button = Button(self, "Play")
        self.sb = ScoreBoard(self)

    def create_alien(self, alien_number, alien_width, number_row, alien_height):
        new_alien = Alien(self)
        new_alien.x = alien_width + 2 * alien_number * alien_width
        new_alien.rect.x = new_alien.x
        new_alien.y = alien_height + 2 * number_row * alien_height
        new_alien.rect.y = new_alien.y
        self.aliens.add(new_alien)

    def create_fleet(self):
        alien = Alien(self)
        # size属性是一个元组，包括了rect属性的宽和高
        alien_width, alien_height = alien.rect.size
        # alien_width = alien.rect.width
        # 计算一行可以容纳多少个外星人
        available_space_x = self.settings.width - (2 * alien_width)
        number_aliens_x = available_space_x // (alien_width * 2)

        # 计算可以容纳多少行外星人
        available_space_y = self.settings.height - (3 * alien_height) - self.ship.rect.height
        number_rows = available_space_y // (2 * alien_height) - 1
        for number_row in range(number_rows):
            for alien_number in range(number_aliens_x):
                self.create_alien(alien_number, alien_width, number_row, alien_height)

    def fire_bullet(self):
        new_bullet = Bullet(self)
        self.bullets.add(new_bullet)

    def check_keydown(self, event):
        if not self.stats.game_Active:
            if event.key == pygame.K_p:
                self.stats.game_Active = True
            elif event.key == pygame.K_q:
                self.end_game()
        else:
            if event.key == pygame.K_q:
                self.end_game()
            elif event.key == pygame.K_RIGHT:
                self.ship.moving_right = True
            elif event.key == pygame.K_LEFT:
                self.ship.moving_left = True
            elif event.key == pygame.K_SPACE:
                self.fire_bullet()

    def check_keyup(self, event):
        if event.key == pygame.K_RIGHT:
            self.ship.moving_right = False
        elif event.key == pygame.K_LEFT:
            self.ship.moving_left = False

    def check_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                self.check_keydown(event)
            elif event.type == pygame.KEYUP:
                self.check_keyup(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 得到鼠标的位置坐标，然后检测是否点击了开始按钮
                mouse_pos = pygame.mouse.get_pos()
                self.check_play_button(mouse_pos)

    def update_event(self):
        self.screen.fill(self.settings.color)
        if not self.stats.game_Active:
            self.play_button.draw_button()
        else:
            self.sb.show_Score()
            self.ship.blitme()
            for bullet in self.bullets.sprites():
                bullet.draw_bullet()
            self.aliens.draw(self.screen)
        pygame.display.flip()

    def check_play_button(self, mouse_pos):
        button_clicked = self.play_button.rect.collidepoint(mouse_pos)
        # 游戏开始后，Play按钮区域不可用
        if button_clicked and not self.stats.game_Active:
            self.start_game()

    def start_game(self):
        # 重置游戏状态
        self.stats.reset_stats()
        self.aliens.empty()
        self.bullets.empty()
        self.create_fleet()
        self.ship.center_ship()
        self.settings.init_dynamic_settings()
        self.sb.prep_ships()
        self.sb.prep_grade()
        self.sb.prep_Score()
        # 让鼠标光标不可见
        pygame.mouse.set_visible(False)

    def end_game(self):
        with open('max_grade.txt', 'w') as file:
            file.write(self.stats.highest_score)
        sys.exit()

    def check_bullet_alien_collisions(self):
        collisions = pygame.sprite.groupcollide(self.bullets, self.aliens, False, True)
        if collisions:
            for aliens in collisions.values():
                # 可能同一个子弹打中了多个外星人，每一个外星人都应该计入分数
                self.stats.score += (self.settings.alien_points * len(aliens))
            self.sb.prep_Score()
            self.sb.check_highest_score()
        if not self.aliens:
            # 如果外星人都被消灭了，就删除现有的子弹并且显示新的外星人
            self.bullets.empty()
            self.create_fleet()
            self.settings.increase_speed()
            self.sb.prep_grade()

    # 检查外星人是否接触到屏幕底端
    def check_aliens_bottom(self):
        for alien in self.aliens.sprites():
            if alien.rect.bottom >= self.screen.get_rect().bottom:
                self.ship_hit()
                break

    def update_bullet(self):
        self.bullets.update()
        # 删除已经消失的子弹（比如飞出屏幕外的子弹）
        for bullet in self.bullets.sprites().copy():
            if bullet.rect.top <= 0:
                self.bullets.remove(bullet)
        self.check_bullet_alien_collisions()

    def change_fleet_direction(self):
        for alien in self.aliens.sprites():
            alien.rect.y += self.settings.fleet_drop_speed
        self.settings.fleet_direction *= -1

    def check_fleet_edges(self):
        for alien in self.aliens.sprites():
            if alien.check_edges():
                self.change_fleet_direction()
                break

    def update_aliens(self):
        # 判断外星人是否与边缘碰撞
        self.check_fleet_edges()
        # 更新外星人移动特性
        self.aliens.update()
        if pygame.sprite.spritecollideany(self.ship, self.aliens):
            self.ship_hit()
        self.check_aliens_bottom()

    # 相应飞船和外星人相撞
    def ship_hit(self):
        if self.stats.ships_left > 0:
            self.stats.ships_left -= 1
            self.sb.prep_ships()
            # 清空剩余的外星人和子弹
            self.aliens.empty()
            self.bullets.empty()
            # 创建新的外星人并将飞船放到屏幕中央
            self.create_fleet()
            self.ship.center_ship()
            # 游戏暂停0.5s
            sleep(0.5)
        else:
            self.stats.game_Active = False
            pygame.mouse.set_visible(True)

    def run_game(self):
        while True:
            self.check_event()
            if self.stats.game_Active:
                self.ship.update()
                self.update_bullet()
                self.update_aliens()

            # todo 游戏结束时弹出消息框----游戏出口

            else:
                self.end_game()
            self.update_event()


ai = AlienInvasion()
ai.run_game()
