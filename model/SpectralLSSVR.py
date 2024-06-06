import typing
from typing_extensions import TypedDict
from .basis import Basis
from .LSSVR import LSSVR
from .utils import to_complex_coeff
import torch
import numpy as np
# model from fourier/chebyshev series
# coeficients are modeled by LSSVRs that are trained on either the input function coefficients or the discretized input function itself
# coeff . basis(x)


class SpectralLSSVR:
    def __init__(self, basis: Basis, modes: int) -> None:
        self.modes = modes
        self.basis = basis
        self.model = LSSVR()

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
        assert (
            f.shape[0] == x.shape[0]
        ), f"f has shape {f.shape} and x has shape {x.shape}, make sure both has the same number of rows (0th dimension)"
        # compute coefficients
        coeff = self.model.predict(f)
        coeff = to_complex_coeff(coeff)

        # compute approximated function
        return np.dot(self.basis.fn(self.modes, x), coeff)


if __name__ == "__main__":
    # TODO: implement tests
    pass