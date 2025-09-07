import abc
import torch
from ..basis import BasisSubType


class Problem(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def generate(
        self,
        basis: type[BasisSubType],
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[BasisSubType, ...]:
        pass

    @abc.abstractmethod
    def spectral_residual(self, u: BasisSubType, *args, **kwargs) -> BasisSubType:
        pass

    @abc.abstractmethod
    def residual(self, u: BasisSubType, *args, **kwargs) -> BasisSubType:
        pass
