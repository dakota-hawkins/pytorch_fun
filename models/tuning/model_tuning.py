import sys
import pathlib

import torch

from torch.utils.data.dataloader import DataLoader


sys.path.insert(
    0, str(pathlib.Path(__file__).parents[1].joinpath("early_adversarial_dropout"))
)


def train_model(
    model: torch.nn.Module,
    config: dict,
    train_laoder: DataLoader,
    test_loader: DataLoader,
) -> torch.nn.Module:
    model = model(**config["model"])
    device = model.get_device()
    model = set_parallel(model)

    criterion = model.loss


def set_parallel(model: torch.nn.Module, device: str) -> torch.nn.Module:
    if "cuda" in device and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model
