import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import torch
from ..basis import Basis, ResType
from .LSSVR import LSSVR
from ..utils import to_complex_coeff, to_real_coeff, get_metrics
from typing import Literal, Union, Callable
from torchmetrics.functional.regression import (
    mean_squared_error,
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
        svr=LSSVR(),
        verbose: Literal["ALL", "SVR", "LITE", False, None] = None,
    ) -> None:
        """
        __init__


        Arguments:
            basis {Basis} -- Basis to use for evaluating the computed function

        Keyword Arguments:
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
        self.svr = svr
        self.svr.verbose = is_svr_verbose

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
        # TODO: add logic for multidimensional functions (2D+)
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
        assert (
            self.basis.coeff_dtype.is_complex and u_coeff.is_complex()
        ), f"u_coeff ({u_coeff.dtype}) and self.basis ({self.basis.coeff_dtype}) must both be either real or complex"
        self.basis.time_dependent = u_time_dependent
        self.modes = Basis.get_modes(u_coeff, u_time_dependent)
        f = f.flatten(1)
        u_coeff = u_coeff.flatten(1)

        self.print(f"modes: {self.modes}")

        if torch.is_complex(u_coeff):
            # TODO: instance should remember if training output samples are complex
            # this info is used to inform the format of the output during evaluation
            u_coeff = to_real_coeff(u_coeff)
        if torch.is_complex(f):
            f = to_real_coeff(f)
        self.svr.fit(f, u_coeff)
        return self

    def test(
        self,
        f: torch.Tensor,
        u_coeff_targets: torch.Tensor,
        res: ResType | None = None,
    ):
        assert (
            self.basis.coeff_dtype.is_complex and u_coeff_targets.is_complex()
        ), f"u_coeff ({u_coeff_targets.dtype}) and self.basis ({self.basis.coeff_dtype}) must both be either real or complex"
        f = f.flatten(1)
        if torch.is_complex(f):
            logger.debug("transform f to real")
            f = to_real_coeff(f)
        u_coeff_preds = self.svr.predict(f)

        if self.basis.coeff_dtype.is_complex:
            u_coeff_preds = to_complex_coeff(u_coeff_preds)
        u_coeff_preds = u_coeff_preds.unflatten(1, u_coeff_targets.shape[1:])

        if self.basis.time_dependent:
            time_shape = u_coeff_targets.shape[1]

            u_preds = (
                self.basis.inv_transform(u_coeff_preds.flatten(0, 1), res=res)
                .unflatten(0, (-1, time_shape))
                .flatten(1)
            )
            u_targets = (
                self.basis.inv_transform(u_coeff_targets.flatten(0, 1), res=res)
                .unflatten(0, (-1, time_shape))
                .flatten(1)
            )

        else:
            u_preds = self.basis.inv_transform(u_coeff_preds, res=res).flatten(1)
            u_targets = self.basis.inv_transform(u_coeff_targets, res=res).flatten(1)

        if self.basis._complex_funcs:
            u_preds = to_real_coeff(u_preds)
            u_targets = to_real_coeff(u_targets)
        else:
            u_preds = u_preds.real
            u_targets = u_targets.real

        metrics = {
            "spectral": get_metrics(
                to_real_coeff(u_coeff_preds.flatten(1)),
                to_real_coeff(u_coeff_targets.flatten(1)),
            ),
            "function value": get_metrics(u_preds, u_targets),
        }

        return metrics

    def inverse(
        self,
        u_coeff: torch.Tensor,
        points: torch.Tensor,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = mean_squared_error,
        epochs=100,
        generator=torch.Generator().manual_seed(42),
        gain=0.2,
        **optimizer_params,
    ):
        # TODO: add logic for multidimensional functions (2D+)
        f_coeff_pred = self.inverse_coeff(
            u_coeff,
            loss_fn=loss_fn,
            epochs=epochs,
            generator=generator,
            gain=gain,
            **optimizer_params,
        )
        f = self.basis.copy()
        f.coeff = f_coeff_pred
        f_pred = f(points)

        return f_pred

    def inverse_coeff(
        self,
        u_coeff: torch.Tensor,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = mean_squared_error,
        epochs=100,
        generator=torch.Generator().manual_seed(42),
        gain=0.05,
        **optimizer_params,
    ):
        # TODO: add logic for multidimensional functions (2D+)
        assert self.svr.sv_x is not None, "SVR has not been trained, no support vectors"
        f_shape = (u_coeff.shape[0], self.svr.sv_x.shape[1])
        complex_coeff = u_coeff.is_complex()
        original_device = u_coeff.device
        u_coeff = to_real_coeff(u_coeff.flatten(1)).to(self.svr.device)

        # inverse problem
        f_coeff_pred = (
            torch.randn(f_shape, generator=generator).to(self.svr.device) * gain
        )
        f_coeff_pred.requires_grad_()
        optim = torch.optim.Adam([f_coeff_pred], **optimizer_params)

        for epoch in range(epochs):
            optim.zero_grad()
            u_coeff_pred = self.svr.predict(f_coeff_pred)
            loss = loss_fn(u_coeff_pred, u_coeff)
            loss.backward()
            optim.step()
        optim.zero_grad()
        f_coeff_pred.requires_grad_(False)
        if complex_coeff:
            f_coeff_pred = to_complex_coeff(f_coeff_pred)
        f_coeff_pred = f_coeff_pred.unflatten(1, self.modes).to(original_device)
        return f_coeff_pred

    def print(
        self,
        *values: object,
        sep: Union[str, None] = " ",
        end: Union[str, None] = "\n",
    ):
        if self.verbose:
            print(*values, sep=sep, end=end)
