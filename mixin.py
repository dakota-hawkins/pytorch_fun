import torch
import torchviz


class MixNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_ = []
        self.epochs_ = []
        self.to(MixNet.get_device())

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        data_loader,
        n_epochs: int = 1000,
        verbose: bool = True,
        verbose_step=200,
        track_loss=False,
    ):
        self.train()
        device = self.get_device()

        for t in range(n_epochs):
            for batch, (X, y) in enumerate(data_loader):
                X, y = X.to(device), y.to(device)

                y_pred = self(X)
                # function call to allow subclass specific loss evaluations
                loss = self.__calculate_loss(y_pred, y)

                # backprop
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if t == 0 or (t + 1) % verbose_step == 0:
                loss_value = loss.item()
                if verbose:
                    print(
                        f"Loss: {loss_value:>7f} [{t + 1:>5d}/{n_epochs:>5d}]",
                        flush=True,
                    )
                if track_loss:
                    self.loss_.append(loss_value)
                    self.epochs_.append(t)
        if verbose:
            print(f"Loss: {loss_value:>7f} [{t + 1:>5d}/{n_epochs:>5d}]", flush=True)

    def __calculate_loss(self, y_pred, y):
        """Private loss function to provide for easy extension"""
        return self.loss(y_pred, y)

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
