import logging

from fire import Fire
from hydra import compose, initialize
from mlflow import (
    create_experiment,
    get_experiment_by_name,
    set_experiment,
    set_tracking_uri,
    start_run,
)

from mlops.consts import CONFIGS_DIR, DEFAULT_CONFIG_NAME
from mlops.train_model import train_model
from mlops.utils import fetch_dataset, get_model_filename


def train(config_name: str = DEFAULT_CONFIG_NAME, download_dataset: bool = True) -> None:
    with initialize(version_base=None, config_path=CONFIGS_DIR):
        cfg = compose(config_name=config_name)
        model_name, tracking_uri = cfg["model_name"], cfg["tracking_uri"]

        cfg = cfg["train"]
        n_epochs, lr, batch_size, device = cfg["n_epochs"], cfg["learning_rate"], cfg["batch_size"], cfg["device"]

    train_root = fetch_dataset(train=True, download=download_dataset)
    valid_root = fetch_dataset(train=False, download=download_dataset)

    model_filename = get_model_filename(model_name)
    logging.info(
        f"Train model: {model_filename}, n_epochs={n_epochs}, lr={lr}, batch_size={batch_size}, device={device}"
    )

    set_tracking_uri(tracking_uri)
    experiment = get_experiment_by_name(model_name)
    experiment_id = create_experiment(model_name) if experiment is None else experiment.experiment_id
    set_experiment(experiment_id=experiment_id)

    with start_run(run_name=model_filename, experiment_id=experiment_id):
        train_model(
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            device_name=device,
            train_root=train_root,
            valid_root=valid_root,
            model_filename=model_filename,
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    Fire(train)
