import torch
from torch import nn


class NetFCN(nn.Module):
    def __init__(self):
        super(NetFCN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=5),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


model = NetFCN()
non_face = torch.load('data/patches_12_new.pt')
face = torch.load('data/train_12.pt')
data = torch.cat((non_face, face), 0)
