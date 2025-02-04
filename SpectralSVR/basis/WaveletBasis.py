import torch
from .__base import Basis


## Wavelet Basis
# TODO: implement wavelet basis https://pywavelets.readthedocs.io/en/latest/ https://pytorch-wavelets.readthedocs.io/en/latest/readme.html
class WaveletBasis(Basis):
    def __init__(self, coeff: torch.Tensor | None = None) -> None:
        super().__init__(coeff)
