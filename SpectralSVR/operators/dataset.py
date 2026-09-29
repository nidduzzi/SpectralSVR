"""OperatorDataset: the common data every operator baseline trains on.

Built from a problem's generated function pair, it holds the input functions
sampled at fixed sensor locations plus the output functions as an evaluable
basis field. Every adapter reads the view it needs from the same object -- so
SpectralSVR (which wants the output coefficients) and DeepONet (which wants the
output sampled at trunk points) draw from one stored field, and a difference
between two rows is a difference of method rather than of data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..basis import Basis, DomainAxis
from ..utils import DeviceInput, resolve_device


def sensor_grid(
    domain: tuple[DomainAxis, ...], n_per_axis: int, device: DeviceInput = None
) -> torch.Tensor:
    """Half-open (periodic) tensor-product grid over ``domain``.

    Parameters
    ----------
    domain : tuple[tuple[float, float], ...]
        Per-axis ``(start, stop)`` intervals.
    n_per_axis : int
        Points per axis; the returned grid has ``n_per_axis ** len(domain)``
        rows.
    device : DeviceInput
        Where the grid is placed; unstated, the process default.

    Returns
    -------
    torch.Tensor
        Shape ``(m, d)`` sensor coordinates, ``m = n_per_axis ** d``.
    """
    placement = resolve_device(device)
    axes = [
        torch.linspace(
            start, stop, n_per_axis + 1, dtype=torch.float64, device=placement
        )[:-1]
        for start, stop in domain
    ]
    mesh = torch.meshgrid(axes, indexing="ij")
    return torch.stack([g.flatten() for g in mesh], dim=-1)


@dataclass(frozen=True)
class OperatorDataset:
    """Input functions at sensors + output functions as an evaluable field.

    Attributes
    ----------
    sensors : torch.Tensor
        Shape ``(m, d)`` sensor coordinates the input functions are sampled at.
    f : torch.Tensor
        Shape ``(N, m)`` real input-function values at the sensors (the operator
        input / DeepONet branch input).
    u : Basis
        The ``N`` output functions as a basis field -- evaluable at arbitrary
        query points and carrying spectral coefficients.
    """

    sensors: torch.Tensor
    f: torch.Tensor
    u: Basis

    def __post_init__(self) -> None:
        if self.f.shape[1] != self.sensors.shape[0]:
            raise ValueError(
                f"inputs are sampled at {self.f.shape[1]} points but the dataset "
                f"carries {self.sensors.shape[0]} sensors"
            )
        if self.f.shape[0] != len(self.u):
            raise ValueError(
                f"{self.f.shape[0]} input functions against {len(self.u)} output "
                "functions; every input needs the output it maps to"
            )

    @property
    def n_functions(self) -> int:
        return self.f.shape[0]

    @classmethod
    def from_fields(
        cls,
        input_field: Basis,
        output_field: Basis,
        n_sensors: int,
    ) -> "OperatorDataset":
        """Sample ``input_field`` at a sensor grid; keep ``output_field`` as-is.

        Parameters
        ----------
        input_field, output_field : Basis
            The operator's input and output functions (which of a problem's
            generated pair is which is the caller's choice).
        n_sensors : int
            Points per axis for the sensor grid.
        """
        sensors = sensor_grid(
            input_field.domain, n_sensors, device=input_field.coeff.device
        )
        f = input_field(sensors).real.to(torch.float64)
        return cls(sensors=sensors, f=f, u=output_field)

    def targets(self, x: torch.Tensor) -> torch.Tensor:
        """Ground-truth ``u`` at each query row, time first where ``u`` varies in time: ``(N, Q)``."""
        if self.u.time_dependent:
            values = type(self.u).evaluate_time_dependent_at(
                self.u.coeff.to(x.device), x, domain=self.u.domain
            )
        else:
            values = self.u(x)
        return values.real.to(torch.float64)


@dataclass(frozen=True)
class Split:
    """A training and a scoring half of one source, disjoint by construction.

    Only :meth:`partition` builds one, so a pair sharing functions cannot be
    stated and an operation handed a ``Split`` needs no check of its own.

    Attributes
    ----------
    train, test : OperatorDataset
        Disjoint halves of the fields ``partition`` was given.
    """

    train: OperatorDataset
    test: OperatorDataset

    @classmethod
    def partition(
        cls,
        input_field: Basis,
        output_field: Basis,
        n_train: int,
        n_sensors: int,
    ) -> "Split":
        """Cut one pair of fields at ``n_train`` into a training and a scoring half."""
        if not 0 < n_train < len(input_field):
            raise ValueError(
                f"n_train must leave both halves non-empty: got {n_train} "
                f"for {len(input_field)} functions"
            )
        return cls(
            train=OperatorDataset.from_fields(
                input_field[:n_train], output_field[:n_train], n_sensors=n_sensors
            ),
            test=OperatorDataset.from_fields(
                input_field[n_train:], output_field[n_train:], n_sensors=n_sensors
            ),
        )
