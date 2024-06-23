import torch
import torchviz
import tempfile
import pathlib

from torch.utils.data.dataloader import DataLoader


class MixNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_loss_ = []
        self.std_loss_ = []
        self.epochs_ = []
        self.input_size_ = 2
        self.to(self.get_device())

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        data_loader: DataLoader,
        n_epochs: int = 1000,
        verbose: bool = True,
        verbose_step=200,
        track_loss=False,
        early_stopping: bool = False,
        test_loader: DataLoader | None = None,
        early_step_size: int = 1,
    ):
        self.train()
        self.verbose_ = verbose
        self.track_loss_ = track_loss
        self.fit_device_ = self.get_device()
        self.batch_loss_ = torch.zeros(len(data_loader))
        self.n_epochs_ = n_epochs
        if not early_stopping:
            self.__fit(data_loader, verbose_step)
        else:
            assert test_loader is not None
            self.__early_stop_fit(
                data_loader, test_loader, verbose_step, early_step_size
            )

    def __log_loss(self, epoch: int, print_loss: bool):
        if self.track_loss_ or self.verbose_:
            loss_std, loss_mean = torch.std_mean(self.batch_loss_)
            self.mean_loss_.append(loss_mean.item())
            self.std_loss_.append(loss_std.item())
            self.epochs_.append(epoch)
            if print_loss and self.verbose_:
                print(
                    f"Loss: {loss_mean.item():>7f} [{epoch + 1:>5d}/{self.n_epochs_:>5d}]",
                    flush=True,
                )

    def __fit(self, data_loader: DataLoader, verbose_step: int):
        for t in range(self.n_epochs_):
            self.__batch_fit(
                data_loader=data_loader, batch_idx=t, verbose_step=verbose_step
            )
            self.__log_loss(epoch=t, print_loss=t == 0 or (t + 1) % verbose_step == 0)

    def __early_stop_fit(
        self,
        training_loader: DataLoader,
        test_loader: DataLoader,
        verbose_step: int,
        step_size: int,
    ):
        best_loss = float("inf")
        temp_dir = tempfile.TemporaryDirectory()
        model_local = pathlib.Path(temp_dir.name).joinpath("best_model.pt")
        self.test_loss_ = []
        self.test_loss_epochs_ = []
        for t in range(self.n_epochs_):
            self.__batch_fit(
                data_loader=training_loader, batch_idx=t, verbose_step=verbose_step
            )
            self.__log_loss(epoch=t, print_loss=t == 0 or (t + 1) % verbose_step == 0)
            if (t % step_size) == 0:
                test_loss = self.test(test_loader)
                self.test_loss_.append(test_loss)
                self.test_loss_epochs_.append(t)
                if test_loss < best_loss:
                    best_loss = test_loss
                self.cache_model(model_local)
        self.load_saved_model(model_local)

    def cache_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model(self, path: str):
        self.load_saved_model(torch.load(path))

    def __batch_fit(
        self, data_loader: DataLoader, batch_idx: int, verbose_step: int
    ) -> torch.Tensor:
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.fit_device_), y.to(self.fit_device_)
            loss = self.__partial_fit(X, y)

            if (
                (self.track_loss_ or self.verbose_)
                and batch_idx == 0
                or (batch_idx + 1) % verbose_step == 0
            ):
                self.batch_loss_[batch] = loss.item()

    def __partial_fit(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self(X)
        # function call to allow bespoke loss evaluations
        loss = self.__calculate_loss(y_pred, y)

        # backprop
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def __calculate_loss(self, y_pred, y):
        """Private loss function to provide for easy extension"""
        return self.loss(y_pred, y)

    def test(self, dataloader):
        n_batches = len(dataloader)
        test_loss = 0
        self.eval()

        device = self.get_device()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                test_loss += self.__calculate_loss(self(X), y).item()
        return test_loss / n_batches

    def draw_network(self):
        X = torch.randn((1, self.input_size_))
        y = self(X)
        return torchviz.make_dot(y.mean(), params=dict(self.named_parameters()))
