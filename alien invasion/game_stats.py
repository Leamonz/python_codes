class GameStats:
    def __init__(self, game):
        self.settings = game.settings
        self.ship_number = 3
        self.reset_stats()
        self.game_Active = False
        # 记录最高得分
        with open('max_grade.txt', 'r') as file:
            self.highest_score = int(file.read())

    def reset_stats(self):
        self.ships_left = self.ship_number
        self.score = 0
