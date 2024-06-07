import torch
import torchviz


class MixNet(torch.nn.Module):
    def __init__(self, rate: float = 1e-3):
        super().__init__()

    def get_device(self) -> str:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, dataloader, n_epochs: int = 1000, verbose: bool = True):
        self.train()
        device = self.get_device()
        size = len(dataloader.dataset) * n_epochs
        for t in range(n_epochs):
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                y_pred = self(X)
                loss = self.loss(y_pred, y)

                # backprop
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (t) % 200 == 0 and verbose:
                loss = loss.item()
                print(f"Loss: {loss:>7f} [{t + 1:>5d}/{n_epochs:>5d}]", flush=True)
        print(f"Loss: {loss:>7f} [{t + 1:>5d}/{n_epochs:>5d}]", flush=True)

    def test(self, dataloader):
        size = len(dataloader.dataset)
        n_batches = len(dataloader)
        self.eval()
        test_score, test_loss = 0, 0
        device = self.get_device()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                score, loss = self.__score(X, y)
                test_score += score
                test_loss += loss
        print(
            f"Test error: Accuracy: {100 * test_score / size:>0.1f}%, "
            f"Avg loss: {test_loss / n_batches:>8f}"
        )

    def __score(self, X: torch.Tensor, y: torch.Tensor):
        """private score function to avoid eval() + no_grad() overhead"""
        y_pred = self(X)
        loss = self.loss(y_pred, y).item()
        score = (y_pred.argmax(1) == y).type(torch.float).sum().item() / len(y)
        return score, loss

    def draw_network(self):
        X = torch.randn((1, 2))
        y = self(X)
        return torchviz.make_dot(y.mean(), params=dict(self.named_parameters()))
