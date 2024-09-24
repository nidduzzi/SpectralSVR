import abc
import torch
from ..basis import Basis


class Problem(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def generate(
        self, basis: Basis, n: int, m: int, *args, **kwargs
    ) -> list[torch.Tensor]:
        pass

    @abc.abstractmethod
    def spectral_residual(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def residual(self, *args, **kwargs) -> torch.Tensor:
        pass
