import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init


class BasicBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(64, 64, 3, padding=1)

        self.bn_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.bn_2 = nn.BatchNorm2d(64)
        self.pool_1 = nn.MaxPool2d(2)
        self.bn_3 = nn.BatchNorm2d(64)

    def forward(self, X):
        Y = X.clone()
        X = self.bn_1(F.relu(self.conv_1(X)))
        X = self.bn_2(F.relu(self.conv_2(X)))
        X += Y

        X = self.bn_3(self.pool_1(F.relu(X)))
        return X
