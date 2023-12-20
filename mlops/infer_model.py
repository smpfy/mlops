from numpy import ndarray
from onnxruntime import InferenceSession

from .utils import create_data_loader


def infer_model(root: str, model_filename: str) -> list[(int, int)]:
    data_loader = create_data_loader(root, batch_size=1, shuffle=False)
    onnx_model = load_onnx_model(model_filename)

    predictions = []
    for x, y in data_loader:
        y_prediction = predict(onnx_model, x.numpy())
        y_true = y.numpy()[0]
        predictions.append((y_true, y_prediction))

    return predictions


def load_onnx_model(filename: str) -> InferenceSession:
    session = InferenceSession(filename, providers=["CPUExecutionProvider"])
    return session


def predict(session: InferenceSession, x: ndarray) -> int:
    onnx_inputs = {session.get_inputs()[0].name: x}
    onnx_outputs = session.run(None, onnx_inputs)
    y = onnx_outputs[0].argmax()
    return y
