import torch
from torch import nn
from torchvision import transforms
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1].joinpath("weight_decay")))
from nets import MnistMLP


class AdversarialMLP(MnistMLP):
    def __init__(
        self,
        rate: float = 1e-3,
        h1_size: int = 784,
        h2_size: int = 392,
        epsilon: float = 0.07,
        alpha: float = 0.5,
    ):
        super().__init__(rate, h1_size, h2_size)
        self.epsilon = epsilon
        self.alpha = alpha
        self.normalize_image_ = transforms.Normalize(0, 1)

    def make_adversarial_examples(
        self, X: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        return X + self.epsilon * gradients.sign()
        # return self.normalize_image_(X + self.epsilon * gradients.sign())

    def __partial_fit(self, X, y):
        X.requires_grad = True
        main_loss = self.__calculate_loss(self(X), y)
        main_loss.backward()
        X_adv = self.make_adversarial_examples(X, X.grad.data)
        adversarial_loss = self.__calculate_loss(self(X_adv), y)

        loss = self.alpha * main_loss + (1 - self.alpha) * adversarial_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss


class DropOutMLP(MnistMLP):
    def __init__(
        self,
        rate: float = 1e-3,
        h1_size: int = 784,
        h2_size: int = 392,
        p_dropout_in=0.5,
        p_dropout_h=0.8,
    ):
        super().__init__(rate, h1_size, h2_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size_, h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.Dropout(p=p_dropout_h),
            nn.Linear(h2_size, 10),
            nn.LogSoftmax(dim=1),  # Negative log-likelihood expects logged input
        )


class AdversarialDropOut(AdversarialMLP, DropOutMLP):
    def __init__(
        self,
        rate: float = 1e-3,
        h1_size: int = 784,
        h2_size: int = 392,
        p_dropout_in=0.5,
        p_dropout_h=0.8,
        epsilon: float = 0.07,
        alpha: float = 0.5,
    ):
        AdversarialMLP.__init__(self, rate, h1_size, h2_size, epsilon, alpha)
        DropOutMLP.__init__(self, rate, h1_size, h2_size, p_dropout_in, p_dropout_h)
