import logging
from pathlib import Path
from shutil import rmtree

from dvc.api import DVCFileSystem
from fire import Fire
from hydra import compose, initialize

from mlops.consts import (
    CONFIGS_DIR,
    DEFAULT_CONFIG_NAME,
    MODELS_DIR,
    RAW_DIR,
    TMP_TRAIN_DIR,
    TRAIN_DIR,
)
from mlops.prepare_dataset import prepare_dataset
from mlops.train_model import train_model


def train(config_name: str = DEFAULT_CONFIG_NAME, download_dataset: bool = True) -> None:
    with initialize(version_base=None, config_path=CONFIGS_DIR):
        cfg = compose(config_name=config_name)
        model_name = cfg["model_name"]
        cfg = cfg["train"]
        n_epochs, lr, batch_size, device = cfg["n_epochs"], cfg["learning_rate"], cfg["batch_size"], cfg["device"]

    root = TMP_TRAIN_DIR if download_dataset else TRAIN_DIR
    rmtree(root, ignore_errors=True)

    if download_dataset:
        logging.info(f"Download train dataset: {root}")
        DVCFileSystem().get(rpath=TRAIN_DIR, lpath=root, recursive=True)
    else:
        logging.info(f"Prepare train dataset: {root}")
        prepare_dataset(raw_dir=RAW_DIR, train_dir=root)

    model_filename = str(Path(MODELS_DIR) / model_name) + ".onnx"
    logging.info(
        f"Train model: {model_filename}, n_epochs={n_epochs}, lr={lr}, batch_size={batch_size}, device={device}"
    )
    train_model(
        n_epochs=n_epochs, lr=lr, batch_size=batch_size, device=device, root=root, model_filename=model_filename
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    Fire(train)
