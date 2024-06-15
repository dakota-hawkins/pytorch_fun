import torch

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path("../").resolve()))
from mixin import MixNet


class XorNet(MixNet):
    def __init__(self, rate: float = 1e-3):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)
        )
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(X)
        return logits
