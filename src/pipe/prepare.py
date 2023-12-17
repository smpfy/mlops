import bz2
import gzip
import struct
from pathlib import Path

import fire
import numpy as np
import torch
from torchvision.io import write_png

from . import consts


def prepare(
    in_dir: str = consts.RAW_DIR, out_train_dir: str = consts.TRAIN_DIR, out_test_dir: str = consts.TEST_DIR
) -> None:
    in_path = Path(in_dir)
    out_train_path = Path(out_train_dir)
    out_test_path = Path(out_test_dir)
    prepare_mnist(in_path, out_train_path, out_test_path)
    # prepare_usps(in_path, out_train_path, out_test_path)


def prepare_mnist(in_path: Path, train_path: Path, test_path: Path) -> None:
    train_images = load_mnist_images(in_path / "train-images-idx3-ubyte.gz")
    train_labels = load_mnist_labels(in_path / "train-labels-idx1-ubyte.gz")
    save_dataset(train_images, train_labels, train_path, "mnist_")

    test_images = load_mnist_images(in_path / "t10k-images-idx3-ubyte.gz")
    test_labels = load_mnist_labels(in_path / "t10k-labels-idx1-ubyte.gz")
    save_dataset(test_images, test_labels, test_path, "mnist_")


def prepare_usps(in_path: Path, train_path: Path, test_path: Path) -> None:
    train_images, train_labels = load_usps_images_and_labels(in_path / "usps.bz2")
    save_dataset(train_images, train_labels, train_path, "usps_")

    test_images, test_labels = load_usps_images_and_labels(in_path / "usps.t.bz2")
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


def load_usps_images_and_labels(path: Path) -> (np.array, np.array):
    with bz2.open(str(path), mode="rb") as f:
        lines = [line.decode().split() for line in f.readlines()]
        data = [[item.split(":")[-1] for item in line[1:]] for line in lines]
        data = np.array(data, dtype=np.float64)
        data = ((data + 1) / 2 * 255).astype(np.uint8)
        images = data.reshape((-1, 16, 16))
        labels = np.array([int(line[0]) - 1 for line in lines])
    return images, labels


def save_dataset(images: np.array, labels: np.array, out_path: Path, prefix: str) -> None:
    for i, (image, label) in enumerate(zip(images, labels)):
        path = out_path / str(label)
        path.mkdir(parents=True, exist_ok=True)
        data = torch.tensor(image).unsqueeze(0)
        filename = str(path / f"{prefix}{i}.png")
        write_png(data, filename=filename, compression_level=0)


if __name__ == "__main__":
    fire.Fire(prepare)
