import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import torch
from ..basis import Basis, ResType
from .LSSVR import LSSVR
from ..utils import to_complex_coeff, to_real_coeff
from typing import Literal, Union, Callable
from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    symmetric_mean_absolute_percentage_error,
    r2_score,
    relative_squared_error,
)
import logging
# model from fourier/chebyshev series
# coeficients are modeled by LSSVRs that are trained on either the input function coefficients or the discretized input function itself
# coeff . basis(x)

logger = logging.getLogger(__name__)


class SpectralSVR:
    def __init__(
        self,
        basis: Basis,
        C=10.0,
        batch_size_func=lambda dims: 2**21 // dims + 1,
        dtype=torch.float32,
        svr=LSSVR,
        verbose: Literal["ALL", "SVR", "LITE", False, None] = None,
        **kwargs,
    ) -> None:
        """
        __init__


        Arguments:
            basis {Basis} -- Basis to use for evaluating the computed function

        Keyword Arguments:
            C {float} -- regularization term with smaller values meaning less complicated models (default: {10.0})
            sigma {float} -- kernel bandwidth (default: {1.0})
            verbose {False | "All" | "SVR" | "lite"} -- verbosity levels, False for no debug logs, All for all logs, LSSVR for logs from LSSVR only, lite all logs except LSSVR (default: {False})
        """
        is_svr_verbose = False
        self.verbose = False
        match verbose:
            case "ALL":
                self.verbose = True
                is_svr_verbose = True
            case "SVR":
                is_svr_verbose = True
            case "LITE":
                self.verbose = True
            case None:
                self.verbose = False

        self.basis = basis
        self.svr = svr(
            C=C,
            batch_size_func=batch_size_func,
            dtype=dtype,
            verbose=is_svr_verbose,
            **kwargs,
        )

    def forward(
        self,
        f: torch.Tensor,
        x: torch.Tensor,
        periods: tuple[float, ...]
        | None = None,  # TODO: use basis args like period etc to make it easier to change for different basis
    ) -> torch.Tensor:
        """
        forward

        Compute F = D(f) and evaluate F(x) where D is an abritrary operator and F is "transformed" function f

        Arguments:
            f {torch.Tensor} -- m discretized input functions to transform using the approximated operator
            x {torch.Tensor} -- m evaluation points for the transformed input functions

        Returns:
            torch.Tensor -- _description_
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        assert (
            x.shape[1] == len(self.modes)
        ), f"Make sure x is in shape (num_points, dimensions) or (num_points,) for 1D. x has shape {x.shape} and modes is {self.modes}"

        # compute coefficients
        if torch.is_complex(f):
            f = to_real_coeff(f)
        coeff = self.svr.predict(f)
        # convert to complex if basis needs complex values so that the reshaping is correct
        if self.basis.coeff_dtype.is_complex:
            coeff = to_complex_coeff(coeff)

        self.print(f"coeff: {coeff.shape}")
        return self.basis.evaluate(
            coeff=coeff.reshape((f.shape[0], *self.modes)),
            x=x,
            periods=periods,
            time_dependent=self.basis.time_dependent,
        )

    def train(
        self, f: torch.Tensor, u_coeff: torch.Tensor, u_time_dependent: bool = False
    ):
        """
        train _summary_

        fit the lssvr to predict the output function coefficients from the input function


        Arguments:
            f {torch.Tensor} -- n flattened input functions
            u_coeff {torch.Tensor} -- n output functions coefficients
            u_u_time_dependent {bool} -- whether the output coefficients are time dependent or not (default: {False})
        """
        self.basis.time_dependent = u_time_dependent
        if self.basis.coeff_dtype.is_complex:
            u_coeff = to_complex_coeff(u_coeff)
        self.modes = Basis.get_modes(u_coeff, u_time_dependent)
        self.print(f"modes: {self.modes}")
        if f.ndim > 2:
            f = f.flatten(1)
        if u_coeff.ndim > 2:
            u_coeff = u_coeff.flatten(1)

        if torch.is_complex(u_coeff):
            u_coeff = to_real_coeff(u_coeff)
        if torch.is_complex(f):
            f = to_real_coeff(f)
        self.svr.fit(f, u_coeff)
        return self

    def test(
        self,
        f: torch.Tensor,
        u_coeff_targets: torch.Tensor,
        res: ResType = 200,
    ):
        if torch.is_complex(f):
            logger.debug("transform f to real")
            f = to_real_coeff(f)
        u_coeff_preds = self.svr.predict(f)
        if torch.is_complex(u_coeff_targets):
            logger.debug("transform u_coeff to real")
            u_coeff_targets = to_real_coeff(u_coeff_targets)

        def get_metrics(preds: torch.Tensor, targets: torch.Tensor):
            nan_pred_sum = torch.isnan(preds).sum().item()
            mse = mean_squared_error(preds, targets)
            rmse = mean_squared_error(preds, targets, squared=False)
            mae = mean_absolute_error(preds, targets)
            r2 = r2_score(preds, targets)
            smape = symmetric_mean_absolute_percentage_error(preds, targets)
            rse = relative_squared_error(preds, targets)
            rrse = relative_squared_error(preds, targets, squared=False)
            return {
                "mse": mse.item(),
                "rmse": rmse.item(),
                "mae": mae.item(),
                "r2": r2.item(),
                "smape": smape.item(),
                "rse": rse.item(),
                "rrse": rrse.item(),
                "pred_nan_sum": nan_pred_sum,
            }

        grid = self.basis.grid(res).flatten(0, -2)

        u_preds = self.basis.evaluate(
            coeff=to_complex_coeff(u_coeff_preds),
            x=grid,
            time_dependent=self.basis.time_dependent,
        ).real
        u_targets = self.basis.evaluate(
            coeff=to_complex_coeff(u_coeff_targets),
            x=grid,
            time_dependent=self.basis.time_dependent,
        ).real

        metrics = {
            "spectral": get_metrics(u_coeff_preds, u_coeff_targets),
            "function value": get_metrics(u_preds, u_targets),
        }

        return metrics

    def inverse(
        self,
        u: torch.Tensor,
        points: torch.Tensor,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = mean_squared_error,
        epochs=100,
        generator=torch.Generator().manual_seed(42),
        gain=0.2,
        **optimizer_params,
    ):
        assert self.svr.sv_x is not None, "SVR has not been trained, no support vectors"
        f_shape = (u.shape[0], self.svr.sv_x.shape[1])
        # inverse problem
        f_pred: torch.Tensor = torch.randn(f_shape, generator=generator) * gain
        f_pred.requires_grad_()

        optim = torch.optim.Adam([f_pred], **optimizer_params)
        for epoch in range(epochs):
            optim.zero_grad()
            u_pred = self.forward(f_pred, points).real
            loss = loss_fn(u_pred, u)
            loss.backward()
            optim.step()
        optim.zero_grad()
        return f_pred

    def print(
        self,
        *values: object,
        sep: Union[str, None] = " ",
        end: Union[str, None] = "\n",
    ):
        if self.verbose:
            print(*values, sep=sep, end=end)
