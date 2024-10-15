import codecs
import json
import typing
import functools
import logging

import torch
import numpy as np

# Inspired by:
# https://github.com/zealberth/lssvr
#
logger = logging.getLogger(__name__)


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


Kernel_Type = typing.Literal[
    "linear", "poly", "rbf", "rbf_unoptimized", "frob", "max", "tri"
]


def linear(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    return torch.mm(x_i, x_j.T)


def poly(
    x_i: torch.Tensor, x_j: torch.Tensor, d: float, gamma22: float
) -> torch.Tensor:
    return (gamma22 * linear(x_i, x_j) + 1) ** d


def rbf(x_i: torch.Tensor, x_j: torch.Tensor, neg_gamma22: float) -> torch.Tensor:
    return torch.exp(neg_gamma22 * torch.cdist(x_i, x_j, p=2.0) ** 2)


def tri(x_i: torch.Tensor, x_j: torch.Tensor, sigma: float) -> torch.Tensor:
    return 1 - torch.cdist(x_i, x_j, p=2.0) / sigma


def frob(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    return torch.norm(x_i.unsqueeze(1) - x_j, p="fro", dim=2)


def max_k(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    return torch.max(x_i.unsqueeze(1) - x_j, dim=2).values


def torch_get_kernel(
    name: Kernel_Type,
    **params,
):
    """The method that returns the kernel function, given the 'kernel'
    parameter.
    """
    match name:
        case "linear":
            return linear
        case "poly":
            return functools.partial(
                poly,
                d=params.get("d", 3.0),
                gamma22=((2 * params.get("sigma", 1.0)) ** -2),
            )
        case "rbf":
            return functools.partial(
                rbf, neg_gamma22=-((2 * params.get("sigma", 1.0)) ** -2)
            )
        case "tri":
            return functools.partial(tri, sigma=params.get("sigma", 1.0))
        case "frob":
            return frob
        case "max":
            return max_k
        case _:
            raise KeyError(f"kernel {name} is not defined, try one of {Kernel_Type}")


class LSSVR(object):
    """A class GPU variation that implements the Least Squares Support
    Vector Machine for regression tasks

    It uses PyTorch pseudo-inverse function to solve the dual optimization
    problem  with ordinary least squares. In multi-output regression problems
    a model is fit for each output.

    # Parameters:
    - min_error: float, default = 0.2
        Constant that control the error threshold of current support vectors, it may vary
        in the set (0, 1). The larger max_error is, the higher error is required for
        current data to stay a support vector.
    - max_error: float, default = 0.8
        Constant that control the error threshold of new support vectors, it may vary
        in the set (0, 1). The larger max_error is, the higher error is required for
        new data to become a support vector.
    - C: float, default = 100.0
        Constant that control the regularization of the model, it may vary
        in the set (0, +infinity). The larger C is, the more
        regularized the model will be.
    - kernel: {'linear', 'poly', 'rbf'}, default = 'rbf'
        The kernel used for the model, if set to 'linear' the model
        will not take advantage of the kernel trick, and the LSSVR maybe only
        useful for linear problems.
    - kernel_params: **kwargs, default = depends on 'kernel' choice
        If kernel = 'linear', these parameters are ignored. If kernel = 'poly',
        'd' is accepted to set the degree of the polynomial, with default = 3.
        If kernel = 'rbf', 'sigma' is accepted to set the radius of the
        gaussian function, with default = 1.

    # Attributes:
    - All hyperparameters of section "Parameters".
    - alpha: tensor of size [1, n_support_vectors] if in single output
             regression and [n_outputs, n_support_vectors] for
             multiclass problems
        Each column is the optimum value of the dual variable for each model
        (we have n_outputs == n_classifiers),
        it can be seen as the weight given to the support vectors
        (sv_x, sv_y). As usually there is no alpha == 0, we have
        n_support_vectors == n_train_samples.
    - b: tensor of size [1] if in single regression and [n_outputs]
         for multi-output problems
        The optimum value of the bias of the model.
    - sv_x: tensor of size [n_support_vectors, n_features]
        The set of the supporting vectors attributes, it has the size
        of the training data.
    - sv_y: tensor of size [n_support_vectors, n_outputs]
        The set of the supporting vectors labels. If the label is represented
        by an array of n elements, the sv_y attribute will have n columns.
    - y_indicies: tensor of size [n_outputs]
        The set of indicies for the output.
    - K: function, default = rbf()
        Kernel function.
    """

    def __init__(
        self,
        C=1.0,
        kernel: Kernel_Type = "rbf",
        verbose=False,
        batch_size_func=lambda dims: 2**21 // dims + 7,
        dtype=torch.float32,
        device: torch.device = torch.device("cpu"),
        **kernel_params,
    ):
        self._device = device

        # Hyperparameters
        self.C = C
        self._kernel: Kernel_Type = kernel
        self._initialized_sigma: bool = False
        self._initialized_kernel: bool = False
        self.verbose = verbose
        self.kernel_params = kernel_params
        self.dtype = dtype
        self.batch_size_func = batch_size_func

        # Model parameters
        self.alpha: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        self.sv_x: torch.Tensor | None = None
        self.sv_y: torch.Tensor | None = None
        self.y_indicies: torch.Tensor | None = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    @property
    def kernel(self):
        return self._kernel

    @property
    def K(self):
        if self.kernel in ["rbf", "poly"] and not self._initialized_sigma:
            if self.kernel_params.get("sigma") is None:
                if self.sv_x is None:
                    self.kernel_params["sigma"] = 1.0
                else:
                    self.kernel_params["sigma"] = self.sv_x.var(0).sum().pow(0.5)
                    self._initialized_sigma = True
            else:
                self._initialized_sigma = True
        if not self._initialized_kernel:
            self._K = torch_get_kernel(self._kernel, **self.kernel_params)
            self._initialized_kernel = True

        return self._K

    def _batched_K(self, x_i: torch.Tensor, x_j: torch.Tensor):
        batch_size_i = self.batch_size_func(x_i.shape[1])
        batch_size_j = self.batch_size_func(x_j.shape[1])
        self.print(f"batch_size_i: {batch_size_i}")
        self.print(f"batch_size_j: {batch_size_j}")
        num_samples_i = x_i.shape[0]
        num_samples_j = x_j.shape[0]
        if num_samples_i <= batch_size_i and num_samples_j <= batch_size_j:
            return self.K(x_i, x_j)
        KXX = torch.empty(
            (num_samples_i, num_samples_j), device=self.device, dtype=self.dtype
        )

        for i in range(0, num_samples_i, batch_size_i):
            i_end = min(i + batch_size_i, num_samples_i)
            for j in range(0, num_samples_j, batch_size_j):
                j_end = min(j + batch_size_j, num_samples_j)
                KXX[i:i_end, j:j_end] = self.K(x_i[i:i_end, :], x_j[j:j_end, :])
        return KXX

    def _optimize_parameters(self, X: torch.Tensor, y_values: torch.Tensor):
        """Helper function that optimizes the dual variables through the
        use of the kernel matrix pseudo-inverse.
        """

        A = torch.empty((X.shape[0] + 1,) * 2, device=self.device, dtype=self.dtype)
        A[1:, 1:] = self._batched_K(X, X)
        # KXX = A[1:, 1:]
        self.print("Omega:")
        self.print(A[1:, 1:])
        A[1:, 1:].diagonal().copy_(
            A[1:, 1:].diagonal()
            + torch.ones(
                (A[1:, 1:].shape[0],), device=self.device, dtype=self.dtype
            ).to()
            / self.C
        )
        self.print("H:")
        self.print(A[1:, 1:])
        A[0, 0] = 0
        A[0, 1:] = 1
        A[1:, 0] = 1
        self.print("A:")
        self.print(A)
        shape = np.array(y_values.shape)
        shape[0] += 1
        B = torch.empty(list(shape), device=self.device, dtype=self.dtype)
        B[0] = 0
        B[1:] = y_values
        self.print("B:")
        self.print(B)

        solution: torch.Tensor = torch.linalg.lstsq(
            A.to(dtype=torch.float), B.to(dtype=torch.float)
        ).solution.to(dtype=self.dtype)
        self.print("S:")
        self.print(solution)

        b = solution[0, :]
        self.print("b:")
        self.print(b)
        alpha = solution[1:, :]
        self.print("alpha:")
        self.print(alpha)

        return (b, alpha)

    def fit(
        self,
        X_arr: typing.Union[np.ndarray, torch.Tensor],
        y_arr: typing.Union[np.ndarray, torch.Tensor],
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

        self.sv_x = X
        self.sv_y = y
        self.y_indicies = torch.arange(0, y.shape[1])

        # if len(self.y_indicies) == 1:  # single regression
        y_values = y

        self.b, self.alpha = self._optimize_parameters(X, y_values)

        return self

    @typing.overload
    def predict(self, X: torch.Tensor) -> torch.Tensor: ...

    @typing.overload
    def predict(
        self, X: np.ndarray[typing.Any, np.dtype[np.float_]]
    ) -> np.ndarray[typing.Any, np.dtype[np.float_]]: ...

    def predict(
        self, X: np.ndarray[typing.Any, np.dtype[np.float_]] | torch.Tensor
    ) -> np.ndarray[typing.Any, np.dtype[np.float_]] | torch.Tensor:
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        assert (
            self.alpha is not None and self.sv_x is not None and self.sv_y is not None
        ), "The model doesn't see to be fitted, try running .fit() method first"
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            X_reshaped_torch = X.reshape(-1, 1) if X.ndim == 1 else X
            X_ = X_reshaped_torch.clone().to(self.device, dtype=self.dtype)
        else:
            X_reshaped_np = X.reshape(-1, 1) if X.ndim == 1 else X
            X_ = torch.from_numpy(X_reshaped_np).to(self.device, dtype=self.dtype)

        self.print(f"X:{X_.shape}")
        self.print(f"sv_x:{self.sv_x.shape}")
        KxX = self._batched_K(X_, self.sv_x)

        self.print("Omega:")
        self.print(KxX)
        y_pred = KxX @ self.alpha + self.b
        self.print("y':")
        self.print(y_pred)
        predictions: np.ndarray[typing.Any, np.dtype[np.float_]] | torch.Tensor
        if is_torch:
            predictions = y_pred.to(X)
        else:
            predictions = y_pred.cpu().numpy()

        return predictions.reshape(-1) if X.ndim == 1 else predictions

    def get_correlation_image(self):
        # using dot product
        assert (
            self.sv_x is not None
        ), "The model needs to be trained first before correlation image can be generated"
        return self._batched_K(self.sv_x, self.sv_x).mm(self.sv_x)

    def get_p_matrix(self):
        assert (
            self.sv_x is not None and self.alpha is not None
        ), "The model needs to be trained first before correlation image can be generated"
        return self.sv_x.T.mm(self.alpha)

    def dump(self, filepath="model", only_hyperparams=False):
        """This method saves the model in a JSON format.
        - filepath: string, default = 'model'
            File path to save the model's json.
        - only_hyperparams: boolean, default = False
            To either save only the model's hyperparameters or not, it
            only affects trained/fitted models.
        """
        model_json = {
            "type": "LSSVR",
            "hyperparameters": {
                "C": self.C,
                "kernel": self._kernel,
                "kernel_params": self.kernel_params,
            },
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json["parameters"] = {
                "alpha": self.alpha,
                "b": self.b,
                "sv_x": self.sv_x,
                "sv_y": self.sv_y,
                "y_indicies": self.y_indicies,
            }

        dump_model(
            model_dict=model_json, file_encoder=torch_json_encoder, filepath=filepath
        )

    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        """This class method loads a model from a .json file.
        - filepath: string
            The model's .json file path.
        - only_hyperparams: boolean, default = False
            To either load only the model's hyperparameters or not, it
            only has effects when the dump of the model as done with the
            model's parameters.
        """
        model_json = load_model(filepath=filepath)

        if model_json["type"] != "LSSVR":
            raise Exception(f"Model type '{model_json['type']}' doesn't match 'LSSVR'")

        lssvr = LSSVR(
            C=model_json["hyperparameters"]["C"],
            kernel=model_json["hyperparameters"]["kernel"],
            **model_json["hyperparameters"]["kernel_params"],
        )

        if (model_json.get("parameters") is not None) and (not only_hyperparams):
            params = model_json["parameters"]
            device = lssvr.device

            lssvr.alpha = torch.Tensor(params["alpha"]).double().to(device)
            lssvr.b = torch.Tensor(params["b"]).double().to(device)
            lssvr.sv_x = torch.Tensor(params["sv_x"]).double().to(device)
            lssvr.sv_y = torch.Tensor(params["sv_y"]).double().to(device)
            lssvr.y_indicies = torch.Tensor(params["y_indicies"]).double().to(device)

        return lssvr

    def print(
        self,
        *values: object,
    ):
        if self.verbose:
            print(*values)
