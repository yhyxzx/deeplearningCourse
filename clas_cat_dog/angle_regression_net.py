import torch.nn as nn
from basic_block import BasicBlock



class AngleRegressionNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Conv2d(3, 64, 1)
        self.block_seq = nn.Sequential(*[BasicBlock() for _ in range(4)])

        self.fc = nn.Sequential(nn.Linear(5184, 1024), nn.ReLU(),nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 32),nn.ReLU(), nn.Linear(32, 2))


    def forward(self, X):
        X = self.encode(X)
        X = self.block_seq(X)

        X = nn.Flatten()(X)
        X = self.fc(X)
        return X
