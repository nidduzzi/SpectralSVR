"""SpectralConfig: the coefficient-independent configuration of a basis.

A basis wears several hats -- a template/spec, a coefficient container, and a
plotter. This value object carves out the *spec* hat: the geometry and
evaluation knobs that do not depend on any particular set of coefficients
(the domain each axis lives on, how that domain is sampled, and the
memory/approximation tuning for transforms).

Keeping it separate means a basis used purely as a template (e.g. a model's
untrained basis, which holds no coefficients) still has a well-defined domain:
domain resolution takes an explicit mode count instead of reading it off the
coefficient tensor's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .domain import (
    DomainAxis,
    DomainInputType,
    domain_lengths,
    domainInputType_to_tuple,
)
from .sampling import SamplingInputType, SamplingScheme, samplings_to_tuple
from .strategy import DEFAULT_EVALUATION_STRATEGY, EvaluationStrategy


@dataclass(frozen=True, kw_only=True)
class SpectralConfig:
    """Coefficient-independent basis configuration (domain, sampling, strategy).

    ``domain`` and ``sampling`` are stored as given -- a single value broadcast to
    every axis, or one per axis -- and resolved to a concrete per-axis tuple only
    on demand, against an explicit mode count. (A domain may be periodic on one
    axis and not another, so node placement is per-axis too.)
    """

    sampling: SamplingInputType
    domain: DomainInputType = None
    strategy: EvaluationStrategy = DEFAULT_EVALUATION_STRATEGY
    coeff_dtype: torch.dtype = torch.complex64

    @property
    def real_dtype(self) -> torch.dtype:
        """The real precision matching ``coeff_dtype``.

        Coefficients are stored complex; the real form is what the interleaved
        bridge produces on the way to a model, so complex is the primary value
        and this is derived from it.
        """
        return torch.empty(0, dtype=self.coeff_dtype).real.dtype

    def resolve_domain(self, modes: tuple[int, ...]) -> tuple[DomainAxis, ...]:
        """Broadcast the raw domain to one ``(start, stop)`` pair per axis."""
        return domainInputType_to_tuple(self.domain, modes)

    def resolve_sampling(
        self, ndim: int, default: SamplingScheme
    ) -> tuple[SamplingScheme, ...]:
        """Broadcast the sampling input to one scheme per axis."""
        return samplings_to_tuple(self.sampling, ndim, default)

    def lengths(self, modes: tuple[int, ...]) -> tuple[float, ...]:
        """Per-axis length ``stop - start`` for the given mode count."""
        return domain_lengths(self.resolve_domain(modes))

    def with_domain(self, domain: DomainInputType) -> "SpectralConfig":
        """A copy with a different domain (this object is immutable)."""
        return replace(self, domain=domain)

    def with_precision(self, coeff_dtype: torch.dtype) -> "SpectralConfig":
        """A copy storing coefficients at a different complex precision.

        Double precision is substantially slower for matrix-bound work on
        consumer hardware -- measured 16.9x on a 2048-square matmul, against
        2.8x for a transform -- and doubles the memory each coefficient
        occupies. Choose it for work whose accuracy demands it, such as an
        anchor whose oracle saturates at single precision, rather than by
        default.
        """
        return replace(self, coeff_dtype=coeff_dtype)
