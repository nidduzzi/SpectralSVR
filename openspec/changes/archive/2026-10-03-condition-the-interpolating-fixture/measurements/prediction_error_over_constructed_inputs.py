"""Measure both tests' prediction error over the constructed inputs, with the fixture frozen as of the recorded commit."""

import itertools
import json
import subprocess
import sys
from pathlib import Path

import torch

from SpectralSVR import FourierBasis, SpectralSVR
from SpectralSVR.model import LSSVR
from SpectralSVR.operators import OperatorDataset, SpectralSVROperator

HOST = torch.device("cpu")
DOMAIN = ((0.0, 1.0), (0.0, 1.0))
UNIT = (0.0, 1.0)
INTERPOLATING_C = 1e6
SPREAD_REPEATS = 8
WORST_FIELDS_RERUN = 30
HERE = Path(__file__).resolve().parent
RECORD = HERE.parents[3] / "tests" / "operators" / "prediction_error_record.json"


def seeded_draw(functions, sensors, seed):
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randn(sensors, functions, generator=generator, dtype=torch.float64)


def orthonormal_inputs(functions, sensors, seed):
    q, r = torch.linalg.qr(seeded_draw(functions, sensors, seed))
    return (q * torch.sign(torch.diagonal(r))).T


def time_varying_field(seed, functions, slices, modes):
    generator = torch.Generator().manual_seed(seed)
    slice_values = torch.randn(functions, slices, modes, generator=generator)
    coeff = FourierBasis.transform(slice_values.flatten(0, 1)).reshape(functions, slices, modes)
    return slice_values, FourierBasis(coeff=coeff, domain=DOMAIN, time_dependent=True)


def time_invariant_field(seed, functions, modes):
    generator = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(functions, (modes, modes), generator=generator)
    return FourierBasis(coeff=coeff, domain=DOMAIN)


def slice_times(slices):
    if slices == 1:
        return torch.full((1,), UNIT[0], dtype=torch.float64)
    return torch.linspace(*UNIT, slices, dtype=torch.float64)


def grid_points(modes):
    return torch.arange(modes, dtype=torch.float64) / modes


def rows(times, space):
    mesh = torch.meshgrid(times, space, indexing="ij")
    return torch.stack([axis.flatten() for axis in mesh], dim=-1)


def dataset_over(u, seed):
    sensors = torch.linspace(0, 1, 4, dtype=torch.float64).unsqueeze(-1)
    return OperatorDataset(sensors=sensors, f=orthonormal_inputs(len(u), 4, seed), u=u)


def interpolating_regressor():
    return LSSVR(kernel="linear", C=INTERPOLATING_C, dtype=torch.float32)


def predictions_test_error(seed, functions, slices, modes):
    slice_values, u = time_varying_field(seed, functions, slices, modes)
    dataset = dataset_over(u, seed)
    query = rows(slice_times(slices), grid_points(modes))
    fitted = SpectralSVROperator(FourierBasis(), interpolating_regressor()).fit(dataset)
    predicted = fitted.predict(dataset.f, query).cpu()
    return ((predicted - slice_values.flatten(1)).abs().max() / slice_values.abs().max()).item()


def state_machine_error(seed, functions, slices, modes, varies_in_time):
    u = (
        time_varying_field(seed, functions, slices, modes)[1]
        if varies_in_time
        else time_invariant_field(seed, functions, modes)
    )
    dataset = dataset_over(u, seed)
    model = SpectralSVR(FourierBasis(), interpolating_regressor())
    model.train(dataset.f, u.coeff, u_time_dependent=u.time_dependent)
    query = rows(slice_times(slices if varies_in_time else modes), grid_points(modes))
    predicted = model.forward(dataset.f, query, domain=DOMAIN).real
    targets = dataset.targets(query)
    return ((predicted - targets).abs().max() / max(targets.abs().max().item(), 1.0)).item()


def measured(metric, cases):
    return [{"metric": metric, **case, "relative_error": error(**case)} for case, error in cases]


def main():
    commit = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    predictions = [
        {"seed": seed, "functions": functions, "slices": slices, "modes": modes}
        for functions, slices, modes in itertools.product(range(1, 5), range(1, 10), range(1, 10))
        for seed in range(10)
    ] + [{"seed": seed, "functions": 4, "slices": 1, "modes": modes} for modes in (1, 2, 3) for seed in range(10, 677)]
    predictions.append({"seed": 2211, "functions": 4, "slices": 1, "modes": 3})
    machine = [
        {"seed": seed, "functions": functions, "slices": slices, "modes": modes, "varies_in_time": varies}
        for functions, slices, modes, varies in itertools.product(range(1, 5), range(1, 10), range(1, 8), (True, False))
        for seed in range(5)
    ]
    records = measured("predictions test", [(c, predictions_test_error) for c in predictions])
    records += measured("state machine", [(c, state_machine_error) for c in machine])
    spread = []
    for metric, error in (("predictions test", predictions_test_error), ("state machine", state_machine_error)):
        worst = sorted((r for r in records if r["metric"] == metric), key=lambda r: -r["relative_error"])
        for record in worst[:WORST_FIELDS_RERUN]:
            case = {k: v for k, v in record.items() if k not in ("metric", "relative_error")}
            repeats = [error(**case) for _ in range(SPREAD_REPEATS)]
            spread.append({"metric": metric, **case, "least": min(repeats), "most": max(repeats)})
    for record in records + spread:
        print(json.dumps(record))
    summary = {
        "commit": commit,
        "fields": {m: sum(r["metric"] == m for r in records) for m in ("predictions test", "state machine")},
        "worst": {m: max(r["relative_error"] for r in records if r["metric"] == m) for m in ("predictions test", "state machine")},
        "worst_over_reruns": {m: max(s["most"] for s in spread if s["metric"] == m) for m in ("predictions test", "state machine")},
        "seed_253_floor": min(s["least"] for s in spread if s["metric"] == "predictions test" and s["seed"] == 253 and s["modes"] == 2),
    }
    RECORD.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
