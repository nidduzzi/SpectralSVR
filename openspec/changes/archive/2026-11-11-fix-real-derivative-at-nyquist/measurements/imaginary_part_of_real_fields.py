import json
import subprocess
import sys

import torch

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.basis.strategy import EvaluationStrategy

FUNCTIONS = 3
SEED = 0
OFF_GRID_POINTS = 64
MODE_COUNTS = ((8,), (9,), (16,), (8, 8), (9, 9), (8, 9))
APPROXIMATE = EvaluationStrategy(memory_budget_mb=1e-9)
EXACT = EvaluationStrategy(allow_approximate=False)


def clean_checkout_commit():
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout
    if changed:
        raise SystemExit(f"tracked files differ from the checked-out commit:\n{changed}")
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def imaginary_over_real(values):
    return float(values.imag.abs().max() / values.real.abs().max())


def readings(field):
    modes = field.modes
    points = torch.rand(OFF_GRID_POINTS, len(modes), generator=torch.Generator().manual_seed(SEED + 1), dtype=torch.float64)
    finer = tuple(slice(0.0, 1.0, 2 * m + 1) for m in modes)
    return {
        "native grid": field.get_values(),
        "off grid, exact": FourierBasis.evaluate(field.coeff, points, strategy=EXACT),
        "off grid, approximate": FourierBasis.evaluate(field.coeff, points, strategy=APPROXIMATE),
        "finer grid": field.get_values(res=finer),
        "first derivative, native grid": field.grad(dim=0, ord=1).get_values(),
        "second derivative, native grid": field.grad(dim=0, ord=2).get_values(),
        "integral, native grid": field.integral(dim=0, ord=1).get_values(),
    }


commit = clean_checkout_commit()
for modes in MODE_COUNTS:
    field = FourierBasis.generate(FUNCTIONS, modes, generator=torch.Generator().manual_seed(SEED))
    for reading, values in readings(field).items():
        json.dump({"commit": commit, "modes": modes, "reading": reading, "imaginary_over_real": imaginary_over_real(values)}, sys.stdout)
        sys.stdout.write("\n")
    if len(modes) == 2:
        time_dependent = field.to_time_dependent(nt=5)
        through_time = FourierBasis.evaluate(
            time_dependent.coeff,
            torch.rand(OFF_GRID_POINTS, 1, generator=torch.Generator().manual_seed(SEED + 2), dtype=torch.float64),
            t=torch.linspace(0, 1, 7, dtype=torch.float64),
            time_dependent=True,
        )
        json.dump({"commit": commit, "modes": modes, "reading": "off grid, through time", "imaginary_over_real": imaginary_over_real(through_time)}, sys.stdout)
        sys.stdout.write("\n")

ANTIDERIVATIVE_DRAWS = 2000
worst = {"even": 0.0, "odd": 0.0}
for draw in range(ANTIDERIVATIVE_DRAWS):
    functions, modes = 1 + draw % 5, 3 + draw % 14
    _, derivative = Antiderivative().generate(
        FourierBasis, functions, modes, generator=torch.Generator().manual_seed(draw), u0=0
    )
    parity = "even" if modes % 2 == 0 else "odd"
    worst[parity] = max(worst[parity], imaginary_over_real(derivative.get_values()))
for parity, ratio in worst.items():
    json.dump({"commit": commit, "reading": f"antiderivative ut, native grid, worst of {ANTIDERIVATIVE_DRAWS} draws, {parity} modes", "imaginary_over_real": ratio}, sys.stdout)
    sys.stdout.write("\n")
