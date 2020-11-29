import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m = nn.Sequential(
            # 3 X 24 X 24
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            # 64 X 24 X 24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 64 X 12 X 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.m(x)
        return x
