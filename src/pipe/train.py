import fire
import torch
import torch.nn as nn
import torch.optim as optim
from hydra import compose, initialize

from . import consts, utils
from .net import Net


def train(train_dir: str = consts.TRAIN_DIR, model_filename: str = consts.MODEL_FILENAME) -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")["train"]
        n_epochs, lr, batch_size, device = cfg["n_epochs"], cfg["learning_rate"], cfg["batch_size"], cfg["device"]
    print(f"n_epochs={n_epochs}, lr={lr}, batch_size={batch_size}, device={device}")

    device = torch.device(device)
    loader = utils.create_data_loader(train_dir, batch_size=batch_size, shuffle=True, drop_last=False)
    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.NLLLoss(reduction="mean")

    model.train()
    for epoch in range(1, n_epochs + 1):
        loss_epoch = 0.0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * x.shape[0]
        loss_epoch /= len(loader.dataset)
        print(f"epoch {epoch}: loss={loss_epoch}")

    x = loader.dataset[0][0][None, :, :, :].to(device)
    utils.save_model_to_onnx(model, x=x, filename=model_filename)


if __name__ == "__main__":
    fire.Fire(train)
