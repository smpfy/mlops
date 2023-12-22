import logging
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from git import Repo
from mlflow import log_metrics, log_params

from .net import Net
from .utils import create_data_loader


def train_model(
    n_epochs: int, lr: float, batch_size: int, device_name: str, train_root: str, valid_root: str, model_filename: str
) -> None:
    device = torch.device(device_name)
    train_loader = create_data_loader(train_root, batch_size=batch_size, shuffle=True)
    valid_loader = create_data_loader(valid_root, batch_size=batch_size, shuffle=False)
    model = Net().to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.NLLLoss(reduction="mean").to(device=device)

    for epoch in range(1, n_epochs + 1):
        train_loss, train_n_samples, train_accuracy = 0.0, 0.0, 0.0
        valid_loss, valid_n_samples, valid_accuracy = 0.0, 0.0, 0.0

        model.train()
        for x, y in train_loader:
            x, y = x.to(device=device), y.to(device=device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_n_samples += x.shape[0]
            train_loss += loss.item() * x.shape[0]
            train_accuracy += (y_pred.argmax(dim=1) == y).sum().item()

        model.eval()
        with torch.inference_mode():
            for x, y in valid_loader:
                x, y = x.to(device=device), y.to(device=device)

                y_pred = model(x)
                loss = criterion(y_pred, y)

                valid_n_samples += x.shape[0]
                valid_loss += loss.item() * x.shape[0]
                valid_accuracy += (y_pred.argmax(dim=1) == y).sum().item()

        metrics = {
            "train_loss": train_loss / train_n_samples,
            "train_accuracy": train_accuracy / train_n_samples,
            "valid_loss": valid_loss / valid_n_samples,
            "valid_accuracy": valid_accuracy / valid_n_samples,
        }
        log_metrics(metrics, step=epoch)

        version = Repo(search_parent_directories=True).git.rev_parse("HEAD")
        params = {"n_epochs": n_epochs, "lr": lr, "batch_size": batch_size, "device": device_name, "version": version}
        log_params(params)

    x = train_loader.dataset[0][0][None, :, :, :].to(device=device)
    save_model_to_onnx(model, x=x, filename=model_filename)


def save_model_to_onnx(model: nn.Module, x: torch.Tensor, filename: str) -> None:
    logging.info(f"Save model: {filename}")

    path = Path(filename)
    path.parents[0].mkdir(parents=False, exist_ok=True)

    output = torch.onnx.dynamo_export(model, x)
    output.save(filename)

    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model, full_check=True)
