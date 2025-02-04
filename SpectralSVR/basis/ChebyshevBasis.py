import torch
from .__base import Basis


## Chebyshev basis
# TODO: implement chebyshev basis https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
class ChebyshevBasis(Basis):
    def __init__(self, coeff: torch.Tensor | None = None) -> None:
        super().__init__(coeff)
