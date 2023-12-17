import fire
from PIL import Image

from . import consts, utils


def infer(path: str, model_filename: str = consts.MODEL_FILENAME) -> int:
    transform = utils.create_image_transforms()
    with Image.open(path) as img:
        x = transform(img)
    onnx_model = utils.load_onnx_model(model_filename)
    y = utils.predict(onnx_model, x.numpy()[None, :, :, :])
    return y


if __name__ == "__main__":
    fire.Fire(infer)
