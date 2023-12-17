import json

import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from onnxruntime import InferenceSession
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def create_image_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )


def create_data_loader(root: str, batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    dataset = ImageFolder(root=root, transform=create_image_transforms())

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=drop_last
    )


def save_model_to_onnx(model: nn.Module, x: torch.tensor, filename: str) -> None:
    output = torch.onnx.dynamo_export(model, x)
    output.save(filename)

    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model, full_check=True)


def load_onnx_model(filename: str) -> InferenceSession:
    session = InferenceSession(filename, providers=["CPUExecutionProvider"])
    return session


def predict(session: InferenceSession, x: np.ndarray) -> int:
    onnx_inputs = {session.get_inputs()[0].name: x}
    onnx_outputs = session.run(None, onnx_inputs)
    y = onnx_outputs[0].argmax()
    return y


def save_metrics(metrics: dict[str, float], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(metrics, f)
