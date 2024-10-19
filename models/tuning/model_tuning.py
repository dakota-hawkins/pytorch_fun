import sys
import pathlib
import pickle
import torch
import typing
import tempfile
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import ray
from ray.train import Checkpoint
from sklearn import metrics

sys.path.insert(
    0, str(pathlib.Path(__file__).parents[1].joinpath("early_adversarial_dropout"))
)


def ray_tune_train_model(
    model_type: torch.nn.Module,
    config: dict,
    training_data: Dataset,
    test_data: Dataset,
    n_epochs: int,
    scorer: typing.Callable,
) -> torch.nn.Module:
    model = model_type(**config["model"])
    device = model.get_device()
    model = set_parallel(model)

    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = pathlib.Path(checkpoint_dir).joinpath("data.pkl")
            with open(data_path, "rb") as f:
                checkpoint_state = pickle.load(f)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            model.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_loader = DataLoader(
        training_data, batch_size=int(config["batch_size"]), shuffle=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    for epoch_steps, epoch in enumerate(range(start_epoch, n_epochs)):
        running_loss = 0.0
        # mini-batch training loop
        for i, (train_X, train_y) in enumerate(train_loader, start=1):
            train_X, train_y = train_X.to_device(device), train_y.to_device(device)

            # zero parameter gradients
            model.optimzer.zero_grad()
            y_pred = model(train_X)
            curr_loss = model.loss(y_pred, train_y)
            curr_loss.backward()
            model.optimizer.step()

            running_loss += curr_loss.item()

            if i % 1000 == 0:
                print(
                    f"Epoch: {epoch_steps}, minibatch {i:5d}: Avg. Epoch Loss: {running_loss / epoch_steps}"
                )
                running_loss = 0.0
        # validation loss
        test_loss = 0.0
        test_score = 0.0
        for test_X, test_y in test_loader:
            with torch.no_grad():
                test_X, test_y = test_X.to_device(device), test_y.to_device(device)

                # predict output and score
                y_pred = model(test_X)
                curr_loss = model.loss(y_pred, test_y)
                test_loss += curr_loss.cpu().numpy()
                test_score += scorer(y_pred, test_y)

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(pathlib.Path(checkpoint_dir) / "data.pkl", "wb") as f:
                pickle.dump(checkpoint_data, f)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            ray.train.report(
                {
                    "loss": test_loss / len(test_data),
                    "score": test_score / len(test_data),
                }
            )


def torch_mcc(y_pred, y):
    return metrics.matthews_corrcoef(y.cpu(), y_pred.cpu())


def set_parallel(model: torch.nn.Module, device: str) -> torch.nn.Module:
    if "cuda" in device and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model
