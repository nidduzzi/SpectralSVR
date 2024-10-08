import torch
from ..basis import Basis
from . import Problem


class Antiderivative(Problem):
    """
    Antiderivative problem for one dimension

    This class defines and generates the antiderivative problem in one dimension. The functions themselves may be multidimensional but the derivative is only in the 0th mode dimension
    """
    def __init__(self) -> None:
        super().__init__()

    def generate(
        self,
        basis: Basis,
        n: int,
        modes: int | list[int],
        u0,
        *args,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> list[Basis]:
        if isinstance(modes, int):
            modes = [modes]
        assert n > 0, "number of samples n must be more than 0"
        for i, m in enumerate(modes):
            assert m > 0, f"number of harmonics m must be more than 0 at dim {i}"
        assert u0 is not None, "integration constant u0 must not be None"
        # generate derivative functions
        ut = basis.generate(n, modes, generator=generator, *args, **kwargs)
        assert (
            ut.coeff is not None
        ), "generated derivative functions ut should not have None coeff"
        ut.coeff[:, 0].mul_(0)
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

        results = [u, ut]
        return results

    def spectral_residual(self, u: Basis, ut: Basis, *args, **kwargs) -> Basis:
        residual = u.grad() - ut
        if residual.coeff is not None:
            residual.coeff[:, 0].mul_(0)

        return residual

    def residual(self, *args, **kwargs) -> Basis:
        residual = super().residual(*args, **kwargs)
        raise NotImplementedError("Function value residual not implemented")
        return residual
