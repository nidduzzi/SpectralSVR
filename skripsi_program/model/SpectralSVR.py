import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import torch
from ..basis import FourierBasis
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
        basis: FourierBasis,
        C=10.0,
        batch_size_func=lambda dims: 2**21 // dims + 1,
        dtype=torch.float32,
        svr=LSSVR,
        verbose: Literal["ALL", "LSSVR", "LITE", False] = False,
        **kwargs,
    ) -> None:
        """
        __init__


        Arguments:
            basis {Basis} -- Basis to use for evaluating the computed function

        Keyword Arguments:
            C {float} -- regularization term with smaller values meaning less complicated models (default: {10.0})
            sigma {float} -- kernel bandwidth (default: {1.0})
            verbose {False | "All" | "LSSVR" | "lite"} -- verbosity levels, False for no debug logs, All for all logs, LSSVR for logs from LSSVR only, lite all logs except LSSVR (default: {False})
        """
        is_lssvr_verbose = False
        self.verbose = False
        match verbose:
            case "ALL":
                self.verbose = True
                is_lssvr_verbose = True
            case "LSSVR":
                is_lssvr_verbose = True
            case "LITE":
                self.verbose = True

        self.basis = basis
        self.svr = svr(
            C=C,
            batch_size_func=batch_size_func,
            dtype=dtype,
            verbose=is_lssvr_verbose,
            **kwargs,
        )

    def forward(
        self,
        f: torch.Tensor,
        x: torch.Tensor,
        periods: list[float]
        | None = None,  # TODO: use basis args like period etc to make it easier to change for different basis
        batched: bool = True,
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
        coeff = to_complex_coeff(coeff)
        scaling = 1.0 / torch.prod(torch.Tensor(self.modes))

        self.print(f"batched: {batched}")
        self.print(f"coeff: {coeff.shape}")
        if batched:
            # coeff_x_basis = coeff.unsqueeze(1) * basis_values.unsqueeze(0)
            # self.print(f"coeff_x_basis: {coeff_x_basis.shape}")
            # sum_coeff_x_basis = coeff_x_basis.flatten(2).sum(2)
            # self.print(f"sum_coeff_x_basis: {sum_coeff_x_basis.shape}")
            return self.basis.evaluate(
                x, coeff.view((-1, *self.modes)), periods, self.modes
            )
        else:
            basis_values = self.basis.fn(x, self.modes, periods=periods)
            self.print(f"basis_values: {basis_values.shape}")
            assert (
                f.shape[0] == x.shape[0]
            ), f"When not batched make sure both has the same number of rows (0th dimension), otherwise use batched in the parameters f has shape {f.shape} and x has shape {x.shape}"
            coeff_x_basis = coeff * basis_values.flatten(1)
            self.print(f"coeff_x_basis: {coeff_x_basis.shape}")
            sum_coeff_x_basis = coeff_x_basis.sum(1, keepdim=True)
            self.print(f"sum_coeff_x_basis: {sum_coeff_x_basis.shape}")

        return scaling * sum_coeff_x_basis

    def train(self, f: torch.Tensor, u_coeff: torch.Tensor, modes: list[int]):
        """
        train _summary_

        fit the lssvr to predict the output function coefficients from the input function


        Arguments:
            f {torch.Tensor} -- n flattened input functions
            u_coeff {torch.Tensor} -- n flattend output functions coefficients
            modes {list[int]} -- list of modes
        """
        assert (
            len(f.shape) == 2
        ), f"f needs to have only 2 dimensions, currently it has shape {f.shape}"
        assert (
            len(u_coeff.shape) == 2
        ), f"u_coeff needs to have only 2 dimensions, currently it has shape {u_coeff.shape}"
        assert (
            torch.prod(torch.Tensor(modes)) == u_coeff.shape[1]
        ), f"modes is {modes} and u_coeff has shape {u_coeff.shape}, the product of modes need to equal to the second dimension of u_coeff"
        self.modes = modes

        if torch.is_complex(u_coeff):
            u_coeff = to_real_coeff(u_coeff)
        if torch.is_complex(f):
            f = to_real_coeff(f)
        self.svr.fit(f, u_coeff)
        return self

    def test(
        self,
        f: torch.Tensor,
        u_coeff: torch.Tensor,
    ):
        if torch.is_complex(f):
            logger.warning("transform f to real")
            f = to_real_coeff(f)
        u_coeff_pred = self.svr.predict(f)
        if torch.is_complex(u_coeff):
            logger.warning("transform u_coeff to real")
            u_coeff = to_real_coeff(u_coeff)
        nan_pred_sum = torch.isnan(u_coeff_pred).sum().item()

        # ssr = ((u_coeff_pred - u_coeff) ** 2).sum(0)
        # mse = ssr.sum() / u_coeff.shape[0]
        # u_coeff_pred can be nan if invalid kernel parameters
        # or data containing nan is input into the model
        # mse.nan_to_num_(3.40282e38)

        # u_coeff_val_mean = u_coeff.mean(dim=0)
        # ssg = ((u_coeff_pred - u_coeff_val_mean) ** 2).sum(0)
        # sst = ((u_coeff - u_coeff_val_mean) ** 2).sum(0)
        # ssr_sst = ssr / sst
        # ssg_sst = ssg / sst

        # r2 = torch.nan_to_num(1 - ssr_sst).sum().item()
        # r2_expected = torch.nan_to_num(1 - ssg_sst).sum().item()
        # u_coeff_pred = to_complex_coeff(u_coeff_pred)
        mse = mean_squared_error(u_coeff_pred, u_coeff)
        rmse = mean_squared_error(u_coeff_pred, u_coeff, squared=False)
        mae = mean_absolute_error(u_coeff_pred, u_coeff)
        r2 = r2_score(u_coeff_pred, u_coeff)
        smape = symmetric_mean_absolute_percentage_error(u_coeff_pred, u_coeff)
        rse = relative_squared_error(u_coeff_pred, u_coeff)
        rrse = relative_squared_error(u_coeff_pred, u_coeff, squared=False)
        metrics = {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
            "r2": r2.item(),
            "r2_abs": r2.abs().item(),
            "smape": smape.item(),
            "rse": rse.item(),
            "rrse": rrse.item(),
            "pred_nan_sum": nan_pred_sum,
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
        # f_pred = torch.empty(f_shape, dtype=u.dtype)  # predicted density
        f_pred: torch.Tensor = torch.randn(f_shape, generator=generator) * gain
        f_pred.requires_grad_()

        # x = torch.arange(0, period, step)
        # pp = torch.meshgrid([x, x], indexing="xy")
        # points = torch.concat([p.flatten().unsqueeze(-1) for p in pp], dim=1)

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
