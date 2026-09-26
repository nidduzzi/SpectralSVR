import logging
from typing_extensions import Self, override

import torch

from ..utils import DEFAULT_REAL_DTYPE, DeviceInput
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from .__base import (
    MultiRegression,
)
from torchmetrics.functional import mean_squared_error

logger = logging.getLogger(__name__)


class FNN(MultiRegression):
    """A class GPU variation that implements the Feed-forward Neural Network for regression tasks


    # Parameters:

    # Attributes:
    - All hyperparameters of section "Parameters".
    """

    def __init__(
        self,
        MAX_EPOCH: int = 100,
        lr: float = 0.001,
        batch_size: int = 4,
        activation: type[nn.Module] = nn.Softplus,
        n_hidden: int = 3,
        w_hidden: int = 100,
        dtype: torch.dtype = DEFAULT_REAL_DTYPE,
        device: DeviceInput = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__(dtype, device, logger)

        self.batch_size: int = batch_size
        self.MAX_EPOCH: int = MAX_EPOCH
        self.lr: float = lr
        self.n_hidden: int = n_hidden
        self.w_hidden: int = w_hidden
        self.activation: type[nn.Module] = activation

        # Model parameters
        self.input: nn.Module | None = None
        self.hidden: nn.Module | None = None
        self.output: nn.Module | None = None
        self.params: nn.Module | None = None
        self._in_features: int | None = None
        self._out_features: int | None = None

    @property
    @override
    def trained(self) -> bool:
        return self.params is not None

    def _build_network(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Sequential(nn.Linear(in_features, self.w_hidden), self.activation()),
            nn.Sequential(
                *sum(
                    [
                        [nn.Linear(self.w_hidden, self.w_hidden), self.activation()]
                        for _ in range(self.n_hidden)
                    ],
                    [],
                )
            ),
            nn.Linear(self.w_hidden, out_features),
        ).to(device=self.device, dtype=self.dtype)

    def _hold_fitted_network(
        self, network: nn.Sequential, in_features: int, out_features: int
    ) -> None:
        self.input, self.hidden, self.output = network
        self._in_features = in_features
        self._out_features = out_features
        self.params = network

    @override
    def _optimize_parameters_and_set(self, X: torch.Tensor, y: torch.Tensor):
        network = self._build_network(X.shape[1], y.shape[1])
        optimizer = torch.optim.Adam(network.parameters(), self.lr)
        network.train()
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, self.batch_size)
        for _ in range(self.MAX_EPOCH):
            for X_batch, y_batch in dl:
                optimizer.zero_grad()
                preds = network.forward(X_batch)
                loss = mean_squared_error(preds, y_batch)
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()

        network.eval()
        network.requires_grad_(False)
        self._hold_fitted_network(network, X.shape[1], y.shape[1])
        return (network,)

    def _predict(self, X_: torch.Tensor):
        assert self.params is not None, (
            "The model doesn't see to be fitted, try running .fit() method first"
        )
        self.params.eval()
        self.logger.debug(f"X:{X_.shape}")
        y_pred = self.params.forward(X_)
        self.logger.debug("y':")
        self.logger.debug(y_pred)
        return y_pred

    @staticmethod
    def _resolve_path(filepath: str) -> str:
        return filepath if filepath.endswith(".pt") else f"{filepath}.pt"

    def dump(self, filepath: str = "model", only_hyperparams: bool = False) -> None:
        """Save the model with ``torch.save`` (weights are not JSON-friendly).
        - filepath: string, default = 'model'
            File path to save the model's ``.pt`` file.
        - only_hyperparams: boolean, default = False
            To either save only the model's hyperparameters or not, it
            only affects trained/fitted models.
        """
        payload: dict[str, object] = {
            "type": "FNN",
            "hyperparameters": {
                "MAX_EPOCH": self.MAX_EPOCH,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "activation": self.activation,
                "n_hidden": self.n_hidden,
                "w_hidden": self.w_hidden,
                "dtype": self.dtype,
            },
        }
        if (self.params is not None) and (not only_hyperparams):
            payload["parameters"] = {
                "in_features": self._in_features,
                "out_features": self._out_features,
                "state_dict": self.params.state_dict(),
            }
        torch.save(payload, self._resolve_path(filepath))

    @classmethod
    def load(cls, filepath: str, only_hyperparams: bool = False) -> Self:
        """Load a model saved by :meth:`dump`.
        - filepath: string
            The model's ``.pt`` file path.
        - only_hyperparams: boolean, default = False
            To either load only the model's hyperparameters or not, it
            only has effects when the dump of the model was done with the
            model's parameters.
        """
        payload = torch.load(cls._resolve_path(filepath), weights_only=False)
        if payload["type"] != "FNN":
            raise Exception(f"Model type '{payload['type']}' doesn't match 'FNN'")

        model = cls(**payload["hyperparameters"])

        params = payload.get("parameters")
        if (params is not None) and (not only_hyperparams):
            network = model._build_network(
                params["in_features"], params["out_features"]
            )
            network.load_state_dict(params["state_dict"])
            network.eval()
            network.requires_grad_(False)
            model._hold_fitted_network(
                network, params["in_features"], params["out_features"]
            )
        return model


NumpyArrayorTensor = np.ndarray | torch.Tensor
