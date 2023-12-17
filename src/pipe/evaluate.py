from pathlib import Path

import fire
from sklearn.metrics import accuracy_score

from . import consts, utils


def evaluate(
    model_filename: str = consts.MODEL_FILENAME, test_dir: str = consts.TEST_DIR, metrics_dir: str = consts.METRICS_DIR
) -> None:
    y_true, y_pred = [], []

    onnx_model = utils.load_onnx_model(model_filename)
    loader = utils.create_data_loader(test_dir, batch_size=1, shuffle=False, drop_last=True)
    for x, y in loader:
        y_true.append(y.numpy()[0])
        y_pred.append(utils.predict(onnx_model, x.numpy()))

    metrics = {"accuracy": accuracy_score(y_true, y_pred, normalize=True)}
    filename = str(Path(metrics_dir) / (Path(model_filename).stem + "_metrics.json"))
    utils.save_metrics(metrics, filename=filename)


if __name__ == "__main__":
    fire.Fire(evaluate)
