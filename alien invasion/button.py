import pygame.font


class Button:
    def __init__(self, game, msg):
        self.screen = game.screen
        self.screen_rect = self.screen.get_rect()
        # 设置按钮的尺寸和颜色、字体信息
        self.width, self.height = 200, 50
        self.button_color = (0, 255, 0)
        self.text_color = (255, 255, 255)
        # 参数1：字体（None表示默认字体）；参数2：字号
        self.font = pygame.font.SysFont(None, 48)
        # 创建按钮的rect对象并使其居中
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.center = self.screen_rect.center
        # 创建按钮标签
        self.prep_msg(msg)

    def prep_msg(self, msg):
        # 将文字渲染为图像然后显示出来
        self.msg_image = self.font.render(msg, True, self.text_color, self.button_color)
        self.msg_image_rect = self.msg_image.get_rect()
        # 让文字居中对齐
        self.msg_image_rect.center = self.rect.center

    def draw_button(self):
        # 在屏幕上绘制一个按钮区域，然后将文本图像放在按钮区域中
        self.screen.fill(self.button_color, self.rect)
        self.screen.blit(self.msg_image, self.msg_image_rect)
