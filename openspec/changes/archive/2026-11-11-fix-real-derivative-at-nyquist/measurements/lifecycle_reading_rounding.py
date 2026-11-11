import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy
import torch
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import invariant, precondition, rule, run_state_machine_as_test

sys.path.insert(0, str(Path.cwd()))
from tests._anchors import fourier_series_at, fourier_series_bound  # noqa: E402
from tests.basis.test_fourier_basis import BasisLifecycle, nodes_of  # noqa: E402

from SpectralSVR import FourierBasis  # noqa: E402
from SpectralSVR.basis._nufft import nufft_available  # noqa: E402
from SpectralSVR.basis.strategy import EvaluationStrategy  # noqa: E402

EXAMPLES = 2000
BUDGETS_MB = (0.0, 2**-10, 2**-8, 2**-6, 2**-4)
OFF_GRID_FRACTIONS = torch.tensor([0.0, 0.137, 0.5, 0.913, 0.999], dtype=torch.float64)
BASIS_MODULE = importlib.import_module("SpectralSVR.basis.FourierBasis")
EXACT = EvaluationStrategy(allow_approximate=False)


def clean_checkout_commit():
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if changed:
        raise SystemExit(
            f"tracked files differ from the checked-out commit:\n{changed}"
        )
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


worst = {}
decisions = {"agree": 0, "disagree": 0}


def record(reading, basis, error, bound):
    if bound == 0.0:
        if error != 0.0:
            raise SystemExit(f"{reading}: error {error} on a zero field")
        return
    ratio = error / bound
    if not math.isfinite(ratio):
        raise SystemExit(f"{reading}: non-finite ratio, error {error}, bound {bound}")
    key = (str(basis.coeff.dtype), reading)
    if ratio > worst.get(key, {}).get("worst", -1.0):
        worst[key] = {
            "worst": ratio,
            "modes": basis.modes,
            "complex_funcs": basis.complex_funcs,
            "domain": basis.domain,
        }


def off_grid_points(domain):
    return torch.stack(
        [start + (stop - start) * OFF_GRID_FRACTIONS for start, stop in domain], dim=1
    )


def max_difference(got, expected):
    return float(numpy.abs(got.detach().cpu().numpy() - expected).max())


def budgeted_and_exact(basis, budget_mb):
    return tuple(
        FourierBasis(
            basis.coeff,
            domain=basis.domain,
            complex_funcs=basis.complex_funcs,
            strategy=strategy,
        )
        for strategy in (EvaluationStrategy(memory_budget_mb=budget_mb), EXACT)
    )


def count_decision(nufft, element_counts, basis, budget_mb):
    over = any(
        count * basis.coeff.element_size() > budget_mb * 2**20
        for count in element_counts
    )
    agrees = (nufft.call_count > 0) == (over and nufft_available())
    decisions["agree" if agrees else "disagree"] += 1


class MeasuredLifecycle(BasisLifecycle):
    @invariant()
    def record_closed_form_readings(self):
        basis = self.basis
        if basis.time_dependent or not self.placeable():
            return
        real = not basis.complex_funcs
        bound = fourier_series_bound(basis.coeff, basis.domain)
        points = off_grid_points(basis.domain)
        off_grid = basis(points)
        record(
            "off grid",
            basis,
            max_difference(
                off_grid,
                fourier_series_at(basis.coeff, points, basis.domain, real=real),
            ),
            bound,
        )
        finer = tuple(2 * m + 1 for m in basis.modes)
        on_finer = basis.get_values(
            res=tuple(
                slice(start, stop, n)
                for n, (start, stop) in zip(finer, basis.domain, strict=True)
            ),
            device=torch.device("cpu"),
        )
        record(
            "finer grid",
            basis,
            max_difference(
                on_finer.flatten(1),
                fourier_series_at(
                    basis.coeff, nodes_of(basis.domain, finer), basis.domain, real=real
                ),
            ),
            bound,
        )
        on_its_grid = torch.fft.ifftn(
            basis.coeff, dim=tuple(range(1, basis.coeff.ndim))
        )
        record(
            "native grid",
            basis,
            float(
                (basis.get_values(device=torch.device("cpu")) - on_its_grid).abs().max()
            ),
            bound,
        )
        if real:
            record(
                "imaginary part off grid",
                basis,
                float(off_grid.imag.abs().max()),
                bound,
            )

    @precondition(lambda self: self.placeable() and not self.basis.time_dependent)
    @rule(budget_mb=st.sampled_from(BUDGETS_MB), points=st.integers(1, 64))
    def record_evaluation_under_a_budget(self, budget_mb, points):
        basis = self.basis
        budgeted, exact = budgeted_and_exact(basis, budget_mb)
        fractions = torch.linspace(0.01, 0.99, points, dtype=torch.float64)
        rows = torch.stack(
            [start + (stop - start) * fractions for start, stop in basis.domain], dim=1
        )
        with patch.object(
            BASIS_MODULE, "nufft_evaluate", wraps=BASIS_MODULE.nufft_evaluate
        ) as nufft:
            values = budgeted(rows)
        count_decision(nufft, (points * math.prod(basis.modes),), basis, budget_mb)
        record(
            "evaluation under a budget",
            basis,
            float((values - exact(rows)).abs().max()),
            fourier_series_bound(basis.coeff, basis.domain),
        )

    @precondition(lambda self: self.placeable() and not self.basis.time_dependent)
    @rule(budget_mb=st.sampled_from(BUDGETS_MB), points=st.integers(2, 24))
    def record_resolution_under_a_budget(self, budget_mb, points):
        basis = self.basis
        budgeted, exact = budgeted_and_exact(basis, budget_mb)
        res = tuple(slice(start, stop, points) for start, stop in basis.domain)
        with patch.object(
            BASIS_MODULE, "nufft_evaluate", wraps=BASIS_MODULE.nufft_evaluate
        ) as nufft:
            values = budgeted.get_values(res=res, device=torch.device("cpu"))
        count_decision(
            nufft,
            tuple(points * m for m in basis.modes if m != points),
            basis,
            budget_mb,
        )
        record(
            "resolution under a budget",
            basis,
            float(
                (values - exact.get_values(res=res, device=torch.device("cpu")))
                .abs()
                .max()
            ),
            fourier_series_bound(basis.coeff, basis.domain),
        )


commit = clean_checkout_commit()
run_state_machine_as_test(
    MeasuredLifecycle,
    settings=settings(
        max_examples=EXAMPLES,
        stateful_step_count=10,
        deadline=None,
        derandomize=True,
        database=None,
        suppress_health_check=list(HealthCheck),
    ),
)
for (coeff_dtype, reading), entry in sorted(worst.items()):
    json.dump(
        {
            "commit": commit,
            "examples": EXAMPLES,
            "coeff_dtype": coeff_dtype,
            "reading": reading,
            **entry,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
json.dump(
    {"commit": commit, "examples": EXAMPLES, "switch decisions": decisions}, sys.stdout
)
sys.stdout.write("\n")
