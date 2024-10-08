import abc
import torch
from ..basis import Basis


class Problem(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def generate(
        self,
        basis: Basis,
        n: int,
        modes: int | list[int],
        *args,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> list[Basis]:
        pass

    @abc.abstractmethod
    def spectral_residual(self, *args, **kwargs) -> Basis:
        pass

    @abc.abstractmethod
    def residual(self, *args, **kwargs) -> Basis:
        pass
