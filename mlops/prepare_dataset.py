import bz2
import gzip
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.io import write_png


def prepare_dataset(raw_dir: str, train_dir: Optional[str] = None, test_dir: Optional[str] = None) -> None:
    raw_path = Path(raw_dir)
    train_path = Path(train_dir) if train_dir is not None else None
    test_path = Path(test_dir) if test_dir is not None else None
    prepare_mnist(raw_path, train_path, test_path)
    # prepare_usps(raw_path, train_path, test_path)


def prepare_mnist(raw_path: Path, train_path: Optional[Path], test_path: Optional[Path]) -> None:
    if train_path is not None:
        train_images = load_mnist_images(raw_path / "train-images-idx3-ubyte.gz")
        train_labels = load_mnist_labels(raw_path / "train-labels-idx1-ubyte.gz")
        save_dataset(train_images, train_labels, train_path, "mnist_")

    if test_path is not None:
        test_images = load_mnist_images(raw_path / "t10k-images-idx3-ubyte.gz")
        test_labels = load_mnist_labels(raw_path / "t10k-labels-idx1-ubyte.gz")
        save_dataset(test_images, test_labels, test_path, "mnist_")


def prepare_usps(raw_path: Path, train_path: Optional[Path], test_path: Optional[Path]) -> None:
    if train_path is not None:
        train_images, train_labels = load_usps_images_and_labels(raw_path / "usps.bz2")
        save_dataset(train_images, train_labels, train_path, "usps_")

    if test_path is not None:
        test_images, test_labels = load_usps_images_and_labels(raw_path / "usps.t.bz2")
        save_dataset(test_images, test_labels, test_path, "usps_")


def load_mnist_images(path: Path) -> np.ndarray:
    with gzip.open(str(path), mode="rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        n_rows, n_cols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
        images = data.reshape((size, n_rows, n_cols))
    return images


def load_mnist_labels(path: Path) -> np.ndarray:
    with gzip.open(str(path), mode="rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
    return labels


def load_usps_images_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with bz2.open(str(path), mode="rb") as f:
        lines = [line.decode().split() for line in f.readlines()]
        pixels = [[item.split(":")[-1] for item in line[1:]] for line in lines]
        data = np.array(pixels, dtype=np.float64)
        data = ((data + 1) / 2 * 255).astype(np.uint8)
        images = data.reshape((-1, 16, 16))
        labels = np.array([int(line[0]) - 1 for line in lines])
    return images, labels


def save_dataset(images: np.ndarray, labels: np.ndarray, out_path: Path, prefix: str) -> None:
    for i, (image, label) in enumerate(zip(images, labels)):
        path = out_path / str(label)
        path.mkdir(parents=True, exist_ok=True)
        data = torch.tensor(image).unsqueeze(0)
        filename = str(path / f"{prefix}{i}.png")
        write_png(data, filename=filename, compression_level=0)
