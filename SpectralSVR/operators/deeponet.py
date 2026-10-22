"""DeepONet baseline via deepxde (torch backend), trained by a named reference recipe.

deepxde is an optional dependency (the ``baselines`` extra), imported on first fit
so the package stays importable without it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from ..utils import DeviceInput, resolve_device
from .dataset import OperatorDataset

INITIALIZER = "Glorot normal"


def _import_deepxde():
    """Import deepxde on the torch backend, leaving the process default device as it was."""
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    default_before_import = torch.get_default_device()
    try:
        import deepxde as dde
        from deepxde.nn.pytorch import DeepONetCartesianProd
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "DeepONetOperator needs the 'baselines' extra: uv sync --extra baselines"
        ) from exc
    finally:
        torch.set_default_device(default_before_import)
    return dde, DeepONetCartesianProd


@dataclass(frozen=True)
class Lu2021DeepxdeExample:
    """ReLU DeepONet of deepxde's antiderivative example and the original DeepONet code.

    Sources: deepxde ``examples/operator/antiderivative_aligned.py``;
    ``lululxvi/deeponet`` ``src/deeponet_pde.py``.
    """

    name: ClassVar[str] = "Lu 2021 / deepxde example"
    source: ClassVar[str] = (
        "deepxde examples/operator/antiderivative_aligned.py; "
        "lululxvi/deeponet src/deeponet_pde.py"
    )
    source_iterations: ClassVar[int] = 50_000
    branch_widths: ClassVar[tuple[int, ...]] = (40, 40)
    trunk_widths: ClassVar[tuple[int, ...]] = (40, 40)
    activation: ClassVar[str] = "relu"
    learning_rate: ClassVar[float] = 1e-3
    minimum_iterations: ClassVar[int] = 1


@dataclass(frozen=True)
class Lu2022Burgers:
    """tanh DeepONet of the fair-comparison code: periodic trunk features, outputs scaled per point.

    Source: ``lu-group/deeponet-fno`` ``src/burgers/deeponet.py``. ``periods`` holds one
    entry per trunk coordinate, the period of a periodic coordinate or ``None`` for one
    passed unchanged. A fitted model answers only at the points it was trained on.
    """

    periods: tuple[float | None, ...]
    name: ClassVar[str] = "Lu 2022 Burgers"
    source: ClassVar[str] = "lu-group/deeponet-fno src/burgers/deeponet.py"
    source_iterations: ClassVar[int] = 500_000
    branch_widths: ClassVar[tuple[int, ...]] = (128,) * 4
    trunk_widths: ClassVar[tuple[int, ...]] = (128,) * 3
    activation: ClassVar[str] = "tanh"
    learning_rate: ClassVar[float] = 1e-3
    harmonics: ClassVar[tuple[int, ...]] = (1, 2)
    decay_intervals: ClassVar[int] = 5
    decay_rate: ClassVar[float] = 0.5
    minimum_iterations: ClassVar[int] = decay_intervals

    def __post_init__(self) -> None:
        periodic = [period for period in self.periods if period is not None]
        if not periodic or not all(period > 0 for period in periodic):
            raise ValueError(
                f"the {self.name} recipe needs at least one periodic coordinate, "
                f"each with a positive period; got periods {self.periods}"
            )

    @property
    def trunk_feature_width(self) -> int:
        return sum(
            1 if period is None else 2 * len(self.harmonics) for period in self.periods
        )

    def trunk_features(self, points: torch.Tensor) -> torch.Tensor:
        columns = []
        for coordinate, period in zip(points.unbind(-1), self.periods, strict=True):
            if period is None:
                columns.append(coordinate)
                continue
            for harmonic in self.harmonics:
                angle = 2 * math.pi * harmonic * coordinate / period
                columns += [torch.cos(angle), torch.sin(angle)]
        return torch.stack(columns, dim=-1)

    def decay(self, iterations: int) -> tuple[str, int, float]:
        return ("inverse time", iterations // self.decay_intervals, self.decay_rate)


DeepONetRecipe = Lu2021DeepxdeExample | Lu2022Burgers


@dataclass(frozen=True)
class PointwiseOutputScale:
    """``outputs * std + mean`` per training point, in deepxde's output-transform signature."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def of(cls, targets: np.ndarray) -> PointwiseOutputScale:
        return cls(
            mean=torch.as_tensor(targets.mean(axis=0), device="cpu"),
            std=torch.as_tensor(targets.std(axis=0), device="cpu"),
        )

    def __call__(self, _inputs: Any, outputs: torch.Tensor) -> torch.Tensor:
        return outputs * self.std.to(outputs.device) + self.mean.to(outputs.device)


@dataclass(frozen=True)
class FittedDeepONet:
    """A trained DeepONet, queried through deepxde's predict on the query's device."""

    name: str
    model: Any
    recipe: DeepONetRecipe
    training_points: torch.Tensor

    @property
    def device(self) -> torch.device:
        return next(self.model.net.parameters()).device

    def predict(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        points = x.to(torch.float64)
        match self.recipe:
            case Lu2022Burgers() if not torch.equal(
                points.cpu(), self.training_points.cpu()
            ):
                raise ValueError(
                    f"a DeepONet trained by the {self.recipe.name} recipe scales its "
                    "outputs per training point, so it answers only at the points it "
                    f"was trained on; got {tuple(x.shape)} points against "
                    f"{tuple(self.training_points.shape)}"
                )
        branch = f.detach().cpu().numpy().astype(np.float32)
        trunk = points.detach().cpu().numpy().astype(np.float32)
        with self.device:
            pred = self.model.predict((branch, trunk))
        return torch.as_tensor(pred, dtype=torch.float64, device=x.device)


class DeepONetOperator:
    """DeepONet (unstacked, Cartesian-product) operator baseline, trained on an accelerator.

    The training targets are the output functions sampled on ``trunk`` points.
    ``device`` must name an accelerator; the host is refused.
    """

    name = "DeepONet"

    def __init__(
        self,
        trunk: torch.Tensor,
        recipe: DeepONetRecipe,
        device: DeviceInput,
        iterations: int = 10_000,
        seed: int = 0,
    ) -> None:
        placement = resolve_device(device)
        if not self.trains_on(placement):
            raise ValueError(
                f"DeepONet trains only on an accelerator (cuda, mps, xpu, ...); got "
                f"{placement}. State an accelerator, e.g. --device cuda"
            )
        if iterations < recipe.minimum_iterations:
            raise ValueError(
                f"the {recipe.name} recipe needs at least {recipe.minimum_iterations} "
                f"iterations; got {iterations}"
            )
        match recipe:
            case Lu2022Burgers(periods=periods) if len(periods) != trunk.shape[1]:
                raise ValueError(
                    f"the {recipe.name} recipe states {len(periods)} trunk "
                    f"coordinates but the trunk points have {trunk.shape[1]}"
                )
        self.trunk = trunk.to(torch.float64)
        self.recipe = recipe
        self.device = placement
        self.iterations = iterations
        self.seed = seed

    @staticmethod
    def trains_on(device: torch.device) -> bool:
        return device.type != "cpu"

    @property
    def fit_covers(self) -> str:
        return f"{self.iterations} Adam iterations of the {self.recipe.name} recipe"

    @property
    def protocol(self) -> str:
        return (
            f"{self.recipe.name} recipe from {self.recipe.source}; "
            f"{self.iterations} iterations against the source's "
            f"{self.recipe.source_iterations}"
        )

    def fit(self, dataset: OperatorDataset) -> FittedDeepONet:
        dde, DeepONetCartesianProd = _import_deepxde()

        branch = dataset.f.detach().cpu().numpy().astype(np.float32)
        trunk = self.trunk.detach().cpu().numpy().astype(np.float32)
        y = dataset.targets(self.trunk).detach().cpu().numpy().astype(np.float32)
        n_sensors = branch.shape[1]

        with self.device:
            data = dde.data.TripleCartesianProd(
                X_train=(branch, trunk), y_train=y, X_test=(branch, trunk), y_test=y
            )
            dde.config.set_random_seed(self.seed)
            match self.recipe:
                case Lu2021DeepxdeExample() as recipe:
                    net = DeepONetCartesianProd(
                        [n_sensors, *recipe.branch_widths],
                        [trunk.shape[1], *recipe.trunk_widths],
                        recipe.activation,
                        INITIALIZER,
                    )
                    decay = None
                case Lu2022Burgers() as recipe:
                    net = DeepONetCartesianProd(
                        [n_sensors, *recipe.branch_widths],
                        [recipe.trunk_feature_width, *recipe.trunk_widths],
                        recipe.activation,
                        INITIALIZER,
                    )
                    net.apply_feature_transform(recipe.trunk_features)
                    net.apply_output_transform(PointwiseOutputScale.of(y))
                    decay = recipe.decay(self.iterations)
            model = dde.Model(data, net)
            model.compile("adam", lr=self.recipe.learning_rate, decay=decay)
            model.train(iterations=self.iterations, display_every=self.iterations)
        return FittedDeepONet(
            name=self.name,
            model=model,
            recipe=self.recipe,
            training_points=self.trunk,
        )
