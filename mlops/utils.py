import logging
from pathlib import Path
from shutil import rmtree
from typing import Optional

import torchvision.transforms as transforms
from dvc.api import DVCFileSystem
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .consts import MODELS_DIR, RAW_DIR, TEST_DIR, TMP_RAW_DIR, TRAIN_DIR
from .prepare_dataset import prepare_dataset


def fetch_dataset(train: bool, download: Optional[bool]) -> str:
    if download:
        if Path(TMP_RAW_DIR).is_dir():
            rmtree(TMP_RAW_DIR)

        logging.info(f"Download dataset: {TMP_RAW_DIR}")
        DVCFileSystem().get(rpath=RAW_DIR, lpath=TMP_RAW_DIR, recursive=True)

    raw_dir = TMP_RAW_DIR if download else RAW_DIR
    root = TRAIN_DIR if train else TEST_DIR

    if Path(root).is_dir():
        rmtree(root)

    logging.info(f"Prepare dataset: {root}")
    if train:
        prepare_dataset(raw_dir=raw_dir, train_dir=root)
    else:
        prepare_dataset(raw_dir=raw_dir, test_dir=root)

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
