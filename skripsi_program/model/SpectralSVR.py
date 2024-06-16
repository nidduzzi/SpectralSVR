import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import torch
from ..basis import Basis
from .LSSVR import LSSVR
from ..utils.fourier import to_complex_coeff, to_real_coeff
from typing import Literal
# model from fourier/chebyshev series
# coeficients are modeled by LSSVRs that are trained on either the input function coefficients or the discretized input function itself
# coeff . basis(x)


class SpectralSVR:
    def __init__(
        self,
        basis: Basis,
        C=10.0,
        sigma=1.0,
        batch_size_func=lambda dims: 2**21 // dims + 7,
        dtype=torch.float32,
        svr = LSSVR,
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
            sigma=sigma,
            batch_size_func=batch_size_func,
            dtype=dtype,
            verbose=is_lssvr_verbose,
            **kwargs,
        )

    def forward(
        self,
        f: torch.Tensor,
        x: torch.Tensor,
        periods: list[int]
        | None = None,  # TODO: use basis args like period etc to make it easier to change for different basis
    ) -> torch.Tensor:
        """
        forward

        Compute F = D(f) and evaluate F(x) where D is an abritrary operator

        Arguments:
            f {torch.Tensor} -- m discretized input functions to transform using the approximated operator
            x {torch.Tensor} -- m evaluation points for the transformed input functions

        Returns:
            torch.Tensor -- _description_
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        assert (
            f.shape[0] == x.shape[0]
        ), f"f has shape {f.shape} and x has shape {x.shape}, make sure both has the same number of rows (0th dimension)"
        # compute coefficients
        if torch.is_complex(f):
            f = to_real_coeff(f)
        coeff = self.svr.predict(f)
        coeff = to_complex_coeff(coeff)

        coeff_x_basis = coeff * self.basis.fn(x, self.modes, periods=periods).flatten(1)
        sum_coeff_x_basis = coeff_x_basis.sum(1, keepdim=True)
        scaling = 1.0 / torch.prod(torch.Tensor(self.modes))
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

    def test(
        self,
        f: torch.Tensor,
        u_coeff: torch.Tensor,
    ):
        if torch.is_complex(f):
            f = to_real_coeff(f)
        u_coeff_pred = self.svr.predict(f)
        u_coeff_pred = to_complex_coeff(u_coeff_pred)
        if self.verbose:
            print("TEST COEFF PRED:")
            print(u_coeff_pred[0, :])
            print("TEST COEFF:")
            print(u_coeff[0, :])
        mse = torch.norm(u_coeff_pred - u_coeff, 2) / u_coeff.shape[0]
        print(f"test coeff mse: {mse}")
