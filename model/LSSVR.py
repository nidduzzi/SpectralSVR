import codecs
import json
import typing

import torch
import numpy as np
from typing_extensions import TypedDict


Kernels = TypedDict(
    "Kernels",
    {
        "linear": typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        "poly": typing.Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
        "rbf": typing.Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    },
)


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


def torch_get_kernel(
    name: typing.Union[
        typing.Literal["linear"], typing.Literal["poly"], typing.Literal["rbf"]
    ],
    **params,
):
    """The method that returns the kernel function, given the 'kernel'
    parameter.
    """

    def linear(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return torch.mm(x_i, torch.t(x_j))

    def poly(
        x_i: torch.Tensor, x_j: torch.Tensor, d: float = params.get("d", 3)
    ) -> torch.Tensor:
        return (torch.mm(x_i, torch.t(x_j)) + 1) ** d

    def rbf(
        x_i: torch.Tensor, x_j: torch.Tensor, sigma: float = params.get("sigma", 1)
    ) -> torch.Tensor:
        return torch.exp(-(torch.norm(x_i[:, None] - x_j, dim=2, p=2) ** 2) / (2 * sigma**2))

    kernels: Kernels = {
        "linear": linear,
        "poly": poly,
        "rbf": rbf,
    }

    if kernels.get(name) is None:
        raise KeyError(
            f"Kernel '{name}' is not defined, try one in the list: "
            f"{list(kernels.keys())}."
        )
    else:
        return kernels[name]


class LSSVR:
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
        min_error=0.2,
        max_error=0.8,
        C=100.0,
        kernel="rbf",
        verbose=False,
        **kernel_params,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.min_error = min_error
        self.max_error = max_error
        self.C = C
        self.kernel_ = kernel
        self.verbose = verbose
        self.kernel_params = kernel_params

        # Model parameters
        self.alpha = None
        self.b = None
        self.sv_x = None
        self.sv_y = None
        self.y_indicies = None

        self.K = torch_get_kernel(kernel, **kernel_params)

    def _optimize_parameters(self, X: torch.Tensor, y_values: torch.Tensor):
        """Helper function that optimizes the dual variables through the
        use of the kernel matrix pseudo-inverse.
        """
        KXX = self.K(X, X)
        self.print("omega:")
        self.print(KXX)
        KXX.diagonal().copy_(
            KXX.diagonal()
            + torch.ones((KXX.shape[0],)).to(self.device, dtype=torch.float32) / self.C
        )
        self.print("H:")
        self.print(KXX)

        A = torch.empty(list(np.array(KXX.shape) + 1)).to(
            self.device, dtype=torch.float32
        )
        A[1:, 1:] = KXX
        A[0, 0] = 0
        A[0, 1:] = 1
        A[1:, 0] = 1
        self.print("A")
        self.print(A)
        shape = np.array(y_values.shape)
        shape[0] += 1
        B = torch.empty(list(shape)).to(self.device, dtype=torch.float32)
        B[0] = 0
        B[1:] = y_values
        self.print("B")
        self.print(B)

        A_cross = torch.t(torch.pinverse(A))
        self.print("A_cross")
        self.print(A_cross)

        solution = torch.mm(A_cross, B)
        self.print("S")
        self.print(solution)

        b = solution[0, :]
        self.print("b")
        self.print(b)
        alpha = solution[1:, :]
        self.print("alpha")
        self.print(alpha)

        return (b, alpha)

    def _filter_simillar(self, X_new_data: torch.Tensor, y_new_data: torch.Tensor):
        # source: https://doi.org/10.1007/s12652-022-03798-w
        if self.sv_x is None and self.sv_y is None:
            return X_new_data, y_new_data

        if self.sv_x is None:
            raise ValueError("sv_x is none")
        if self.sv_y is None:
            raise ValueError("sv_y is none")
        if X_new_data.shape[1] != self.sv_x.shape[1]:
            raise ValueError("old data and new data X has mismatched feature size")
        if y_new_data.shape[1] != self.sv_y.shape[1]:
            raise ValueError("old data and new data y has mismatched output size")

        # current_X = self.sv_x.clone().requires_grad_()
        current_X = self.sv_x
        current_pred = self.predict(current_X)
        current_err = torch.norm(current_pred - self.sv_y, p=2, dim=1)
        # current_err.sum().backward(retain_graph=True)

        # new_X = X_new_data.clone().requires_grad_()
        new_X = X_new_data
        new_pred = self.predict(new_X)
        new_err = torch.norm(new_pred - y_new_data, p=2, dim=1)

        if not isinstance(current_err, torch.Tensor):
            raise ValueError(f"current_err ({current_err}) is not a tensor")
        if not isinstance(new_err, torch.Tensor):
            raise ValueError(f"new_err ({new_err}) is not a tensor")
        # new_err.sum().backward(retain_graph=True)

        # similarity_mat = torch.norm(current_data[:, None] - new_data, dim=2, p=2)
        # similar_mat = similarity_mat >= self.error_threshold

        # no_similar_new_data = torch.sum(similar_mat, dim=1) == 0
        # valid_support_vectors = current_data[no_similar_new_data, :]
        # most_similar_new_data = torch.argmin(similarity_mat, dim=1)
        # valid_new_data = new_data[torch.unique(most_similar_new_data), :]
        current_err_range = (
            (current_err.min(), current_err.max())
            if current_err.shape[0] > 0
            else (torch.tensor(0.0), torch.tensor(0.0))
        )
        current_valid = current_err > (
            (current_err_range[1] - current_err_range[0]) * self.min_error
            + current_err_range[0]
        )
        new_err_range = (
            (new_err.min(), new_err.max())
            if new_err.shape[0] > 0
            else (torch.tensor(0.0), torch.tensor(0.0))
        )
        new_valid = new_err > (
            (new_err_range[1] - new_err_range[0]) * self.max_error + new_err_range[0]
        )
        # current_X_grad = current_X.grad.flatten()
        # current_err_d_min, current_err_d_max = current_X_grad.min(), current_X_grad.max()
        # current_valid = current_X_grad > (
        #     (current_err_d_max - current_err_d_min) * self.min_error_percentile + current_err_d_min
        # )
        # new_X_grad = new_X.grad.flatten()
        # new_err_d_min, new_err_d_max = new_X_grad.min(), new_X_grad.max()
        # new_valid = new_X_grad > (
        #     (new_err_d_max - new_err_d_min) * self.max_error_percentile + new_err_d_min
        # )

        # self.print(current_valid.shape)
        # self.print(new_valid.shape)
        self.print(
            f"current_err_max: {current_err_range[1]} ({torch.argmax(current_err)}), current_err_min: {current_err_range[0]} ({torch.argmin(current_err)})"
        )
        # self.print(f"current_err_d_max: {current_err_d_max} ({torch.argmax(current_X_grad)}), current_err_d_min: {current_err_d_min} ({torch.argmin(current_X_grad)})")
        # self.print(current_err[current_valid])
        # self.print(current_X_grad[current_valid])
        self.print(
            f"new_err_max: {new_err_range[1]} ({torch.argmax(new_err)}), new_err_min: {new_err_range[0]} ({torch.argmin(new_err)})"
        )
        # self.print(f"new_err_d_max: {new_err_d_max} ({torch.argmax(new_X_grad)}), new_err_d_min: {new_err_d_min} ({torch.argmin(new_X_grad)})")
        # self.print(new_err[new_valid])
        # self.print(new_X_grad[new_valid])
        self.print(
            f"current_valid: {torch.sum(current_valid)/current_err.shape[0]}, new_valid: {torch.sum(new_valid)/new_err.shape[0]}"
        )

        return torch.vstack(
            (self.sv_x[current_valid, :], X_new_data[new_valid, :])
        ), torch.vstack((self.sv_y[current_valid, :], y_new_data[new_valid, :]))

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
        X_reshaped = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
        if isinstance(X_reshaped, torch.Tensor):
            X = X_reshaped.clone().to(self.device, dtype=torch.float32)
        else:
            X = torch.from_numpy(X_reshaped).to(self.device, dtype=torch.float32)

        y_reshaped = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
        if isinstance(y_reshaped, torch.Tensor):
            y = y_reshaped.clone().to(self.device, dtype=torch.float32)
        else:
            y = torch.from_numpy(y_reshaped).to(self.device, dtype=torch.float32)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X_arr and y_arr does not have the same shape along the 0th dim: (X: "
                + str(X.shape)
                + ", y: "
                + str(y.shape)
                + ")"
            )
        if update:
            X, y = self._filter_simillar(X, y)

        self.sv_x = X
        self.sv_y = y
        self.y_indicies = torch.arange(0, y.shape[1])

        # if len(self.y_indicies) == 1:  # single regression
        y_values = y

        self.b, self.alpha = self._optimize_parameters(X, y_values)

        # else:  # multi-output regression
        #     n_output = len(self.y_indicies)
        #     self.b = torch.empty(n_output, dtype=X.dtype, device=self.device)
        #     self.alpha = torch.empty(n_output, len(y), dtype=X.dtype, device=self.device)
        #     for i in range(n_output):
        #         y_values = y

        #         self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)

    def predict(self, X_arr: typing.Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try running .fit() method first"
            )
        if isinstance(X_arr, torch.Tensor):
            X_reshaped_torch = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
            X = X_reshaped_torch.clone().to(self.device, dtype=torch.float32)
        else:
            X_reshaped_np = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
            X = torch.from_numpy(X_reshaped_np).to(self.device, dtype=torch.float32)

        self.print(f"X:{X.shape}")
        self.print(f"sv_x:{self.sv_x.shape}")
        KxX = self.K(X, self.sv_x)

        # if len(self.y_indicies) == 1:  # binary classification
        # y_values = self.sv_y
        self.print("Omega")
        self.print(KxX)
        y = KxX @ self.alpha + self.b
        self.print("v")
        self.print(y)
        predictions = y

        # else:  # multiclass classification, ONE-VS-ALL APPROACH
        #     y = torch.empty((len(self.y_indicies), len(X)), dtype=X.dtype, device=self.device)
        #     for i in range(len(self.y_indicies)):
        #         # y_values = self.sv_y

        #         y[i] = KxX @ self.alpha[i] + self.b[i]

        #     predictions = y
        # y_pred_labels = torch.stack([self.y_indicies[i] for i in predictions])

        return predictions.reshape(-1) if X_arr.ndim == 1 else predictions

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
                "kernel": self.kernel_,
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
        sep: typing.Union[str, None] = " ",
        end: typing.Union[str, None] = "\n",
    ):
        if self.verbose:
            print(*values, sep=sep, end=end)
