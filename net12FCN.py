import torch
from torch import nn


class NetFCN(nn.Module):
    def __init__(self):
        super(NetFCN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=4),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
