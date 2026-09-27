"""SpectralSVR as an :class:`Operator` (the method under comparison)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..basis import Basis, DomainInputType
from ..model import SpectralSVR
from ..model.__base import MultiRegression
from .dataset import OperatorDataset


@dataclass(frozen=True)
class FittedSpectralSVR:
    """A trained SpectralSVR pipeline, evaluated at query points."""

    name: str
    model: SpectralSVR
    output_domain: DomainInputType

    def predict(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(f, x, domain=self.output_domain).real.to(
            torch.float64
        )


class SpectralSVROperator:
    """Adapter exposing the SpectralSVR pipeline through the Operator interface.

    Trains the LSSVR to map sensor-sampled input functions to the output spectral
    coefficients; the fitted value evaluates the reconstructed output.
    """

    name = "SpectralSVR"

    def __init__(self, basis: Basis, regressor: MultiRegression) -> None:
        self._basis = basis
        self._regressor = regressor

    def fit(self, dataset: OperatorDataset) -> FittedSpectralSVR:
        model = SpectralSVR(self._basis, self._regressor)
        model.train(
            dataset.f, dataset.u.coeff, u_time_dependent=dataset.u.time_dependent
        )
        return FittedSpectralSVR(
            name=self.name, model=model, output_domain=dataset.u.domain
        )
