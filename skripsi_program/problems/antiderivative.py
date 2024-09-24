import torch
from ..basis import Basis
from . import Problem


class Antiderivative(Problem):
    def __init__(self) -> None:
        super().__init__()

    def generate(
        self, basis: Basis, n: int, m: int, u0, *args, **kwargs
    ) -> list[torch.Tensor]:
        assert n > 0, "number of samples n must be more than 0"
        assert m > 0, "number of harmonics m must be more than 0"
        assert u0 is not None, "integration constant u0 must not be None"
        # generate derivative functions
        ut = basis.generate(n, m, *args, **kwargs)
        # compute antiderivative functions
        u = ut.integral()
        # set the integration coefficient
        assert (
            u.coeff is not None and ut.coeff is not None
        ), "Something went wrong, because generated functions have value of None for coeff"
        if isinstance(u0, complex):
            if u.coeff.is_complex():
                u.coeff[:, 0] = torch.tensor(u0)
            else:
                u.coeff[:, 0] = torch.tensor(u0).real
        elif isinstance(u0, float) or isinstance(u0, int):
            if u.coeff.is_complex():
                u.coeff[:, 0] = torch.tensor(u0 + 0j)
            else:
                u.coeff[:, 0] = torch.tensor(u0)
        else:
            u.coeff[:, 0] = u0

        results = super().generate(basis, n, m)
        results = [u.coeff, ut.coeff]
        return results

    def spectral_residual(self, *args, **kwargs) -> torch.Tensor:
        residual = super().spectral_residual(*args, **kwargs)
        return residual

    def residual(self, *args, **kwargs) -> torch.Tensor:
        residual = super().residual(*args, **kwargs)
        return residual
