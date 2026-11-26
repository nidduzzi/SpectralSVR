"""Operator baselines: a common function->function interface.

These wrap neural-operator methods (DeepONet, and later FNO/SNO/NSM) behind one
interface so they can be compared against SpectralSVR on the same problems. An
``Operator`` learns a mapping from an input function (sampled at fixed sensor
locations) to an output function it can then evaluate at arbitrary query points.

The two protocols below are the two states an operator can be in, and ``fit`` is
the only transition between them. Adapters satisfy them structurally and inherit
nothing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .dataset import OperatorDataset


@runtime_checkable
class FittedOperator(Protocol):
    """A trained operator, evaluable at arbitrary query points.

    ``name`` labels the operator in comparison tables and keeps this shape
    distinct from the regressors, which carry ``predict`` without a name.
    """

    @property
    def name(self) -> str:
        """Label for this operator's row in a comparison table."""
        ...

    @property
    def device(self) -> torch.device:
        """Where the fitted state lives."""
        ...

    def predict(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned operator.

        Parameters
        ----------
        f : torch.Tensor
            Shape ``(N, m)`` input functions sampled at the ``m`` training
            sensor locations.
        x : torch.Tensor
            Shape ``(Q, d)`` query points to evaluate each output function at,
            which need not lie on the training grid.

        Returns
        -------
        torch.Tensor
            Shape ``(N, Q)`` predicted output values ``u(x)``.
        """
        ...


@runtime_checkable
class Operator(Protocol):
    """An untrained function-to-function map.

    It carries no prediction method: :meth:`fit` returns the
    :class:`FittedOperator` that does, so predicting before fitting is rejected
    where the call is written rather than by a check inside it.
    """

    @property
    def name(self) -> str:
        """Label for this operator's row in a comparison table."""
        ...

    @property
    def fit_covers(self) -> str:
        """The work one fit does, so a fit time is read against what it measures."""
        ...

    @property
    def protocol(self) -> str:
        """The configuration this operator trains by, its source, and any departure from it."""
        ...

    def fit(self, dataset: OperatorDataset) -> FittedOperator:
        """Train on ``dataset``; returns the fitted operator."""
        ...


# Comparison metrics are the project's existing suite -- ``utils.get_metrics``
# (mse / rmse / mae / r2 / smape / rse / rrse). The benchmark scores every
# operator's predictions with it, so all methods share the same yardstick.
