import matplotlib.pyplot as plt
import numpy as np
import torch
import torchfile
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.fcn = nn.Sequential(
            nn.Linear(16 * 4 * 4, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.fcn(x.view(x.size(0), -1))
        return x


model = Net()
o = torchfile.load('../data/aflw/aflw_12.t7')
faces = np.array(list(o.values()))
