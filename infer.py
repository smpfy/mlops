import logging
from csv import writer as csv_writer

from fire import Fire
from hydra import compose, initialize

from mlops.consts import CONFIGS_DIR, DEFAULT_CONFIG_NAME
from mlops.infer_model import infer_model
from mlops.utils import fetch_dataset, get_model_filename


def infer(config_name: str = DEFAULT_CONFIG_NAME, download_dataset: bool = True) -> None:
    with initialize(version_base=None, config_path=CONFIGS_DIR):
        cfg = compose(config_name=config_name)
        model_name = cfg["model_name"]

    root = fetch_dataset(train=False, download=download_dataset)

    model_filename = get_model_filename(model_name)
    logging.info(f"Infer model: {model_filename}")
    predictions = infer_model(root=root, model_filename=model_filename)

    predictions_filename = model_name + "_infer.csv"
    logging.info(f"Save predictions: {predictions_filename}")
    save_predictions_to_csv(predictions, filename=predictions_filename)


def save_predictions_to_csv(predictions: list[tuple[int, int]], filename: str) -> None:
    with open(filename, mode="w") as file:
        writer = csv_writer(file)
        writer.writerow(["y_true", "y_pred"])
        writer.writerows(predictions)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    Fire(infer)
