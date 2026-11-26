import sys

import numpy
import torch

sys.path.insert(0, "tests/utils")

from SpectralSVR.utils import get_metrics
from test_metric_anchors import (
    DEGENERATE_SPREAD,
    MINIMUM_ERROR,
    TARGET_RAMP,
    metrics_by_numpy,
)

DRAWS = 20000

if __name__ == "__main__":
    rng = numpy.random.default_rng(0)
    absolute: dict[str, float] = {}
    relative: dict[str, float] = {}
    for _ in range(DRAWS):
        rows, columns = int(rng.integers(2, 21)), int(rng.integers(1, 6))
        t = (
            rng.uniform(-10, 10, (rows, columns))
            + TARGET_RAMP * numpy.arange(1, rows + 1)[:, None]
        )
        error = rng.uniform(MINIMUM_ERROR, 10, (rows, columns)) * rng.choice(
            [-1, 1], (rows, columns)
        )
        p = t + error
        if ((t - t.mean(0)) ** 2).sum(0).min() <= DEGENERATE_SPREAD:
            continue
        if numpy.linalg.norm(t, axis=1).min() <= DEGENERATE_SPREAD:
            continue
        measured = get_metrics(torch.from_numpy(p), torch.from_numpy(t))
        for name, expected in metrics_by_numpy(p, t).items():
            difference = abs(measured[name] - expected)
            absolute[name] = max(absolute.get(name, 0.0), difference)
            relative[name] = max(
                relative.get(name, 0.0), difference / (1 + abs(expected))
            )
    for name in absolute:
        print(
            f"{name}: worst absolute {absolute[name]:.3g}, worst against one plus the magnitude {relative[name]:.3g}"
        )
