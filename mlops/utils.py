import logging
from pathlib import Path
from shutil import rmtree
from typing import Optional

import torchvision.transforms as transforms
from dvc.api import DVCFileSystem
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .consts import (
    MODELS_DIR,
    RAW_DIR,
    TEST_DIR,
    TMP_TEST_DIR,
    TMP_TRAIN_DIR,
    TRAIN_DIR,
)
from .prepare_dataset import prepare_dataset


def fetch_dataset(train: bool, download: Optional[bool]) -> str:
    cache_dir = TRAIN_DIR if train else TEST_DIR
    tmp_dir = TMP_TRAIN_DIR if train else TMP_TEST_DIR
    root = tmp_dir if download else cache_dir

    if Path(root).is_dir():
        rmtree(root)

    if download:
        logging.info(f"Download dataset: {root}")
        DVCFileSystem().get(rpath=cache_dir, lpath=root, recursive=True)
        return root

    logging.info(f"Prepare dataset: {root}")
    if train:
        prepare_dataset(raw_dir=RAW_DIR, train_dir=root)
    else:
        prepare_dataset(raw_dir=RAW_DIR, test_dir=root)

    return root


def create_data_loader(root: str, batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    dataset = ImageFolder(root=root, transform=transform)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=drop_last
    )


def get_model_filename(model_name: str) -> str:
    return str(Path(MODELS_DIR) / model_name) + ".onnx"
