import torch
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
        X + self.epsilon * gradients.sign()

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
