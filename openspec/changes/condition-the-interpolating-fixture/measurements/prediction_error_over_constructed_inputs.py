import itertools
import json
import sys

import torch

sys.path.insert(0, ".")
from SpectralSVR import FourierBasis
from SpectralSVR.operators import SpectralSVROperator
from tests._time_varying import grid_points, rows, slice_times, time_varying_field
from tests.operators.test_time_varying_scoring import DOMAIN, dataset_over, interpolating_regressor

HOST = torch.device("cpu")


def relative_error(seed, functions, slices, modes):
    slice_values, coeff = time_varying_field(seed, functions, slices, modes, HOST)
    u = FourierBasis(coeff=coeff, domain=DOMAIN, time_dependent=True)
    dataset = dataset_over(u, seed)
    query = rows(slice_times(slices, HOST), grid_points(modes, HOST))
    fitted = SpectralSVROperator(FourierBasis(), interpolating_regressor()).fit(dataset)
    predicted = fitted.predict(dataset.f, query).cpu()
    scale = slice_values.abs().max()
    return ((predicted - slice_values.flatten(1)).abs().max() / scale).item()


grid = [
    (seed, functions, slices, modes)
    for functions, slices, modes in itertools.product(range(1, 5), range(1, 10), range(1, 10))
    for seed in range(10)
]
corner = [(seed, 4, 1, modes) for modes in (1, 2, 3) for seed in range(10, 677)] + [(2211, 4, 1, 3)]
for seed, functions, slices, modes in grid + corner:
    print(json.dumps({"seed": seed, "functions": functions, "slices": slices, "modes": modes,
                      "relative_error": relative_error(seed, functions, slices, modes)}))
