import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 192, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# LeakyRELU

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        layers = []

        for config in architecture_config[1:]:
            if type(config) == tuple:
                layers.append(
                    ConvBlock(self.in_channels, config[1], kernel_size=config[0], stride=config[2], padding=config[3]))
                self.in_channels = config[1]
            elif type(config) == list:
                for _ in range(config[-1]):
                    layers.append(
                        ConvBlock(self.in_channels, config[0][1], kernel_size=config[0][0], stride=config[0][2],
                                  padding=config[0][3]))
                    self.in_channels = config[0][1]
                    layers.append(
                        ConvBlock(self.in_channels, config[1][1], kernel_size=config[1][0], stride=config[1][2],
                                  padding=config[1][3]))
                    self.in_channels = config[1][1]
            elif config == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.model = nn.Sequential(*layers)
        self.flatten = nn.Flatten(start_dim=1)
        self.fcs = self.create_fcs(**kwargs)

    def create_fcs(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Linear(1024 * split_size * split_size, 496),
            nn.LeakyReLU(0.1),
            nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5))
        )

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        return self.fcs(x)


def test():
    x = torch.randn((1, 3, 448, 448)).to('cuda')
    model = YOLOv1(split_size=7, num_classes=20, num_boxes=2).to('cuda')
    print(model)
    print(model(x).shape)


if __name__ == "__main__":
    test()
