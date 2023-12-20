import onnx
import torch
import torch.nn as nn
import torch.optim as optim

from .net import Net
from .utils import create_data_loader


def train_model(n_epochs: int, lr: float, batch_size: int, device: str, root: str, model_filename: str) -> None:
    device = torch.device(device)
    data_loader = create_data_loader(root, batch_size=batch_size, shuffle=True)
    model = Net().to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.NLLLoss(reduction="mean")

    model.train()
    for epoch in range(1, n_epochs + 1):
        loss_epoch = 0.0
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device=device), y.to(device=device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * x.shape[0]
        loss_epoch /= len(data_loader.dataset)
        print(f"epoch {epoch}: loss={loss_epoch}")

    x = data_loader.dataset[0][0][None, :, :, :].to(device=device)
    save_model_to_onnx(model, x=x, filename=model_filename)


def save_model_to_onnx(model: nn.Module, x: torch.tensor, filename: str) -> None:
    output = torch.onnx.dynamo_export(model, x)
    output.save(filename)

    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model, full_check=True)
