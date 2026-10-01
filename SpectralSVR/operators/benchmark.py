"""Run a set of operators on the same data and tabulate the comparison."""

from __future__ import annotations

import time
from collections.abc import Sequence

import pandas as pd
import torch

from ..utils import DEFAULT_METRICS, get_metrics
from .__base import Operator
from .dataset import Split

RUN_KEY = ["dataset", "operator", "seed"]
SUMMARISED_COLUMNS = ["fit_seconds", *DEFAULT_METRICS]


def benchmark(
    operators: Sequence[Operator],
    split: Split,
    query: torch.Tensor,
) -> pd.DataFrame:
    """Fit each operator on ``split.train`` and score it on ``split.test`` at ``query``.

    Every operator is scored with the project's shared metric suite
    (:func:`utils.get_metrics`) against the same targets, so the columns are
    directly comparable across methods. ``fit_seconds`` brackets ``fit`` alone,
    excluding dataset construction and prediction. An operator that raises is
    not caught: a failure stops the run rather than dropping a row.

    Parameters
    ----------
    operators : Sequence[Operator]
        The baselines to compare (e.g. ``SpectralSVROperator``, ``DeepONetOperator``).
    split : Split
        The training and scoring halves of one source. ``Split.partition`` is the
        only way to build one, so they share no functions by construction.
    query : torch.Tensor
        Shape ``(Q, d)`` points to evaluate the predicted output functions at.

    Returns
    -------
    pandas.DataFrame
        One row per operator: ``operator``, ``fit_seconds``, ``fit_covers``,
        ``fit_device``, then every metric from :func:`utils.get_metrics`.
    """
    target_on_host = split.test.targets(query).detach().cpu().contiguous()
    rows = []
    for op in operators:
        start = time.perf_counter()
        fitted = op.fit(split.train)
        fit_seconds = time.perf_counter() - start
        prediction_on_host = (
            fitted.predict(split.test.f, query).detach().cpu().contiguous()
        )
        rows.append(
            {
                "operator": fitted.name,
                "fit_seconds": fit_seconds,
                "fit_covers": op.fit_covers,
                "fit_device": str(fitted.device),
                **get_metrics(prediction_on_host, target_on_host),
            }
        )
    return pd.DataFrame(rows).set_index("operator")


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Reduce a long table of runs to a centre and a spread per dataset and operator.

    ``runs`` holds one row per ``dataset``, ``operator`` and ``seed``; a repeated
    key is refused rather than weighted twice. Each of ``fit_seconds`` and the
    metric suite becomes ``<column>_mean`` and ``<column>_std`` (sample standard
    deviation, undefined for one seed), beside ``n_seeds`` and the ``seeds`` used.
    """
    repeated = runs.duplicated(subset=RUN_KEY, keep=False)
    if repeated.any():
        raise ValueError(
            "a run table holds each dataset, operator and seed once; repeated: "
            f"{runs.loc[repeated, RUN_KEY].drop_duplicates().values.tolist()}"
        )
    grouped = runs.groupby(["dataset", "operator"], sort=False)
    spread = grouped[SUMMARISED_COLUMNS].agg(["mean", "std"])
    spread.columns = [f"{column}_{statistic}" for column, statistic in spread.columns]
    seeds = grouped["seed"].agg(
        n_seeds="nunique",
        seeds=lambda values: str(sorted(int(seed) for seed in values)),
    )
    return seeds.join(spread)
