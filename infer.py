import logging
from csv import writer as csv_writer
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
    TEST_DIR,
    TMP_TEST_DIR,
)
from mlops.infer_model import infer_model
from mlops.prepare_dataset import prepare_dataset


def infer(config_name: str = DEFAULT_CONFIG_NAME, download_dataset: bool = True) -> None:
    with initialize(version_base=None, config_path=CONFIGS_DIR):
        cfg = compose(config_name=config_name)
        model_name = cfg["model_name"]

    root = TMP_TEST_DIR if download_dataset else TEST_DIR
    rmtree(root, ignore_errors=True)

    if download_dataset:
        logging.info(f"Download test dataset: {root}")
        DVCFileSystem().get(rpath=TEST_DIR, lpath=root, recursive=True)
    else:
        logging.info(f"Prepare test dataset: {root}")
        prepare_dataset(raw_dir=RAW_DIR, test_dir=root)

    path = Path(MODELS_DIR) / model_name
    model_filename = str(path) + ".onnx"
    logging.info(f"Infer model: {model_filename}")
    predictions = infer_model(root=root, model_filename=model_filename)

    predictions_filename = str(path) + "_predictions.csv"
    logging.info(f"Save predictions: {predictions_filename}")
    save_predictions_to_csv(predictions, filename=predictions_filename)


def save_predictions_to_csv(predictions: list[(int, int)], filename: str) -> None:
    with open(filename, mode="w") as file:
        writer = csv_writer(file)
        writer.writerow(["y_true", "y_prediction"])
        writer.writerows(predictions)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    Fire(infer)
