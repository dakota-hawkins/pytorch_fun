import torch
from torch import nn
from torchvision import transforms, datasets

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path("../../").resolve()))
from mixin import MixNet


def get_mnist(train=True):
    data_dir = pathlib.Path("../../").resolve().joinpath("data")
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
    return datasets.MNIST(root=data_dir, transform=tx, train=train)


class MnistMLP(MixNet):
    def __init__(self, rate=1e-3, h1_size=784, h2_size=392):
        super().__init__()
        self.input_size_ = 28 * 28
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size_, h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.Linear(h2_size, 10),
            nn.LogSoftmax(dim=1),  # Negative log-likelihood expects logged input
        )
        self.loss = torch.nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=rate)

    def forward(self, X):
        X = torch.flatten(X, 1)
        logits = self.mlp(X)
        return logits


class L2WeightDecayMnistMLP(MnistMLP):
    def __init__(self, rate=1e-3, h1_size=784, h2_size=392, weight_decay=0.01):
        super().__init__(rate, h1_size, h2_size)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=rate, weight_decay=weight_decay
        )
        self.weight_decay = weight_decay


class RegularizedMnistMLP(MnistMLP):
    def __init__(self, norm=2, rate=1e-3, h1_size=784, h2_size=392, weight_decay=0.01):
        super().__init__(rate, h1_size, h2_size)
        self.weight_decay = torch.tensor(weight_decay)
        self.norm = norm

    def __calculate_loss(self, y_pred, y):
        unweighed_loss = super().__calculate_loss(y_pred, y)
        # don't penalize bias parameters per p. 223 of Goodfellow et al.
        weight_norm = sum(
            torch.linalg.norm(p, self.norm)
            for (name, p) in self.named_parameters()
            if "bias" not in name
        )
        return unweighed_loss + self.weight_decay * weight_norm
