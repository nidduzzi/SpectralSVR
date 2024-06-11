import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import torch
from ..basis import Basis, FourierBasis
from .LSSVR import LSSVR
from ..utils.fourier import to_complex_coeff, to_real_coeff
# model from fourier/chebyshev series
# coeficients are modeled by LSSVRs that are trained on either the input function coefficients or the discretized input function itself
# coeff . basis(x)


class SpectralLSSVR:
    def __init__(self, basis: Basis, C=10.0, sigma=1.0, verbose=False) -> None:
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
            case "All":
                self.verbose = True
                is_lssvr_verbose = True
            case "LSSVR":
                is_lssvr_verbose = True
            case "lite":
                self.verbose = True

        self.basis = basis
        self.lssvr = LSSVR(C=C, sigma=sigma, verbose=is_lssvr_verbose)

    def forward(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
        coeff = self.lssvr.predict(f)
        coeff = to_complex_coeff(coeff)

        basis_values = self.basis.fn(x, x.shape[1], self.modes)
        # compute approximated function
        return (
            1 / (self.modes) * torch.sum(coeff * basis_values, dim=-1).unsqueeze(-1)
        )  # dot product

    def train(self, df: torch.utils.data.dataset.TensorDataset):
        """
        train

        fit the lssvr to predict the output function coefficients from the input function

        Arguments:
            df {torch.utils.data.dataset.TensorDataset} -- Training dataset
        """
        f, u, u_coeff = df[:]
        self.modes = u_coeff.shape[1]

        if torch.is_complex(u_coeff):
            u_coeff = to_real_coeff(u_coeff)
        if torch.is_complex(f):
            f = to_real_coeff(f)
        self.lssvr.fit(f, u_coeff)

    def test(self, df: torch.utils.data.dataset.TensorDataset):
        f, u, u_coeff = df[:]
        if torch.is_complex(f):
            f = to_real_coeff(f)
        u_coeff_pred = self.lssvr.predict(f)
        u_coeff_pred = to_complex_coeff(u_coeff_pred)
        if self.verbose:
            print("TEST COEFF PRED:")
            print(u_coeff_pred[0, :])
            print("TEST COEFF:")
            print(u_coeff[0, :])
        mse = torch.norm(u_coeff_pred - u_coeff, 2) / u_coeff.shape[0]
        print(f"test coeff mse: {mse}")
