import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(16 * 4 * 4, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.linear(x.view(x.size(0), -1))
        return x


model = Net()
non_face = torch.load('data/patches_12_new.pt')
face = torch.load('data/train_12.pt')
data = torch.cat((non_face, face), 0)
