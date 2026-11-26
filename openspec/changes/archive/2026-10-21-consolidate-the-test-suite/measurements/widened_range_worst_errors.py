import itertools
import math
import sys
import warnings

import numpy
import torch

for directory in ("tests", "tests/problems", "tests/utils", "tests/basis"):
    sys.path.insert(0, directory)

from SpectralSVR import FourierBasis
from SpectralSVR.problems import Burgers
from SpectralSVR.utils import (
    _etdrk4_weights,
    resize_modes,
    as_semilinear_solver,
    etdrk4_solver,
    implicit_adams_solver,
)

from _anchors import hermitian_error, hermitian_spectrum
from test_problems import (
    DOMAIN_2D,
    STIFFNESS_CEILING,
    analytic_periodic_pair,
    continuous_burgers_residual,
)
from test_solvers import _closed_form_weights


def manufactured_residuals():
    worst = {"mms": 0.0, "analytic": 0.0}
    for modes, nu in itertools.product(
        (6, 8, 12, 16, 24, 32, 40), (0.01, 0.05, 0.1, 0.2)
    ):
        for seed in (3, 11):
            u, f = Burgers().mms_solution(
                FourierBasis,
                1,
                (modes, modes),
                nu,
                DOMAIN_2D,
                generator=torch.Generator().manual_seed(seed),
            )
            worst["mms"] = max(worst["mms"], continuous_burgers_residual(u, f, nu))
        if modes >= 10:
            u, f = analytic_periodic_pair(modes, nu)
            worst["analytic"] = max(
                worst["analytic"], continuous_burgers_residual(u, f, nu)
            )
    return worst


def burgers_agreement():
    worst, unconverged, runs = 0.0, 0, 0
    for nu, modes in itertools.product((0.005, 0.0125, 0.02), (6, 9, 12)):
        fewest = max(
            401, math.ceil(nu * (math.pi * modes) ** 2 / STIFFNESS_CEILING) + 1
        )
        for steps, seed in itertools.product(
            sorted({fewest, max(801, fewest)}), (7, 11, 13, 17, 19)
        ):

            def run(solver):
                u, _ = Burgers().generate(
                    FourierBasis,
                    2,
                    (modes, modes),
                    u0="random",
                    f=0,
                    nu=nu,
                    time_domain=slice(0, 1, steps),
                    generator=torch.Generator().manual_seed(seed),
                    solver=solver,
                )
                return u.coeff.cpu()

            exponential = run(etdrk4_solver)
            with warnings.catch_warnings(record=True) as raised:
                warnings.simplefilter("always")
                adams = run(as_semilinear_solver(implicit_adams_solver))
            runs += 1
            if raised:
                unconverged += 1
                continue
            worst = max(
                worst,
                float((exponential - adams).abs().max() / exponential.abs().max()),
            )
    return worst, unconverged, runs


def resize_sequences():
    rng = numpy.random.default_rng(0)
    worst_by_count = {}
    for case in range(400):
        modes = int(rng.integers(2, 17))
        source = hermitian_spectrum(modes, seed=case)
        current = source
        for _ in range(int(rng.integers(1, 25))):
            current = resize_modes(
                current, current.shape[1] + int(rng.integers(0, 41)), rescale=False
            )
            returned = resize_modes(current, modes, rescale=False)
            error = max(
                float((returned - source).abs().max()), hermitian_error(current)
            )
            bucket = 10 ** len(str(current.shape[1] // modes))
            worst_by_count[bucket] = max(worst_by_count.get(bucket, 0.0), error)
    return dict(sorted(worst_by_count.items()))


def etdrk4_weight_errors():
    rng = numpy.random.default_rng(0)
    filtered = unfiltered = 0.0
    for _ in range(4000):
        h = float(rng.uniform(0.01, 0.5))
        frequencies = rng.uniform(1.0, 200.0, int(rng.integers(1, 7))) * rng.choice(
            [-1, 1], 1
        )
        frequencies = rng.uniform(1.0, 200.0, len(frequencies)) * rng.choice(
            [-1, 1], len(frequencies)
        )
        linear = 1j * torch.tensor(frequencies, dtype=torch.float64).to(
            torch.complex128
        )
        contour = _etdrk4_weights(linear, h, contour_points=32)
        exact = _closed_form_weights(h * linear, h)
        error = max(float((w - r).abs().max()) for w, r in zip(contour, exact))
        if h * min(abs(frequencies)) >= 0.1:
            filtered = max(filtered, error)
        else:
            unfiltered = max(unfiltered, error)
    return filtered, unfiltered


if __name__ == "__main__":
    print("manufactured residuals (bound 5e-6):", manufactured_residuals())
    print(
        "burgers worst relative, unconverged, runs (bound 1e-4):", burgers_agreement()
    )
    print("resize worst error by count/modes bucket (bound 1e-14):", resize_sequences())
    print(
        "etdrk4 weights worst, at or above and below the filter (bound 1e-10):",
        etdrk4_weight_errors(),
    )
