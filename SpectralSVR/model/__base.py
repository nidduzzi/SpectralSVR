import codecs
import json
import typing
from typing_extensions import Self
import abc
import torch
import numpy as np

NumpyArrayorTensor = np.ndarray | torch.Tensor


def torch_json_encoder(obj):
    if type(obj).__module__ == torch.__name__:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f"""Unable to  "jsonify" object of type :', {type(obj)}""")


def dump_model(model_dict, file_encoder, filepath="model"):
    with open(f"{filepath.replace('.json', '')}.json", "w") as fp:
        json.dump(model_dict, fp, default=file_encoder)


def load_model(filepath="model"):
    helper_filepath = filepath if filepath.endswith(".json") else f"{filepath}.json"
    file_text = codecs.open(helper_filepath, "r", encoding="utf-8").read()
    model_json = json.loads(file_text)

    return model_json


class MultiRegression(abc.ABC):
    def __init__(
        self,
        verbose=False,
        dtype=torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self._device = device
        self.verbose = verbose
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def trained(self)->bool:
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    @abc.abstractmethod
    def _optimize_parameters_and_set(
        self, X: torch.Tensor, y_values: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        pass

    def fit(
        self,
        X_arr: NumpyArrayorTensor,
        y_arr: NumpyArrayorTensor,
        update=False,
    ):
        """Fits the model given the set of X attribute vectors and y labels.
        - X: ndarray or tensor of shape (n_samples, n_attributes)
        - y: ndarray or tensor of shape (n_samples,) or (n_samples, n_outputs)
            If the label is represented by an array of n_outputs elements, the y
            parameter must have n_outputs columns.
        """
        # converting to tensors and passing to GPU
        if isinstance(X_arr, torch.Tensor):
            X = X_arr.to(self.device, dtype=self.dtype)
        else:
            X = torch.from_numpy(X_arr).to(self.device, dtype=self.dtype)
        X = X.view(-1, 1) if X.ndim == 1 else X

        if isinstance(y_arr, torch.Tensor):
            y = y_arr.to(self.device, dtype=self.dtype)
        else:
            y = torch.from_numpy(y_arr).to(self.device, dtype=self.dtype)
        y = y.view(-1, 1) if y.ndim == 1 else y

        assert (
            X.shape[0] == y.shape[0]
        ), f"X_arr and y_arr does not have the same shape along the 0th dim: (X: {X.shape}, y: {y.shape})"

        self._optimize_parameters_and_set(X, y)

        return self

    @abc.abstractmethod
    def _predict(self, X_: torch.Tensor) -> torch.Tensor:
        pass

    @typing.overload
    def predict(self, X: torch.Tensor) -> torch.Tensor: ...

    @typing.overload
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict(
        self,
        X: NumpyArrayorTensor,
    ) -> NumpyArrayorTensor:
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            X_reshaped_torch = X.reshape(-1, 1) if X.ndim == 1 else X
            X_ = X_reshaped_torch.clone().to(self.device, dtype=self.dtype)
        else:
            X_reshaped_np = X.reshape(-1, 1) if X.ndim == 1 else X
            X_ = torch.from_numpy(X_reshaped_np).to(self.device, dtype=self.dtype)

        y_pred = self._predict(X_)
        predictions: np.ndarray | torch.Tensor
        if is_torch:
            predictions = y_pred.to(X)
        else:
            predictions = y_pred.cpu().numpy()

        return predictions.reshape(-1) if X.ndim == 1 else predictions

    @abc.abstractmethod
    def dump(self, filepath="model", only_hyperparams=False):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath, only_hyperparams=False) -> Self:
        pass

    def print(
        self,
        *values: object,
    ):
        if self.verbose:
            print(*values)
