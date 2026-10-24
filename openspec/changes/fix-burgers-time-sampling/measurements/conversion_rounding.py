"""Largest distance between a manufactured pair converted by ``to_time_dependent`` and the pair at its inclusive slice times, scaled by the larger of its largest value and one, over the test's ranges."""

import itertools
import json
import subprocess
import sys

import torch

from SpectralSVR import FourierBasis
from SpectralSVR.problems import Burgers

FUNCTIONS = 2
NU = 0.05
UNIT = (0.0, 1.0)
DOMAIN = (UNIT, UNIT)
SLICE_COUNTS = range(2, 41)
MODE_COUNTS = range(4, 17)
PRECISIONS = (torch.complex64, torch.complex128)


def manufactured_pair(modes, seed):
    return Burgers().mms_solution(
        FourierBasis, FUNCTIONS, modes, NU, DOMAIN, generator=torch.Generator().manual_seed(seed)
    )


def at_precision(field, coeff_dtype):
    return FourierBasis.from_config(
        field.coeff.to(coeff_dtype), field.config.with_precision(coeff_dtype)
    )


def slice_points(slices, space_points):
    times = torch.linspace(*UNIT, slices, dtype=torch.float64)
    space = UNIT[0] + torch.arange(space_points, dtype=torch.float64) * (UNIT[1] - UNIT[0]) / space_points
    mesh = torch.meshgrid(times, space, indexing="ij")
    return torch.stack([axis.flatten() for axis in mesh], dim=-1)


def relative_rounding(manufactured, slices):
    expected = manufactured(slice_points(slices, manufactured.modes[1]))
    converted = manufactured.to_time_dependent(nt=slices).get_values().flatten(1)
    return float((converted - expected).abs().max() / expected.abs().max().clamp_min(1.0))


commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
worst = {}
for seed, (slices, time_modes, space_modes) in enumerate(
    itertools.product(SLICE_COUNTS, MODE_COUNTS, MODE_COUNTS)
):
    pair = manufactured_pair((time_modes, space_modes), seed)
    for coeff_dtype, (name, field) in itertools.product(PRECISIONS, zip(("u", "f"), pair, strict=True)):
        rounding = relative_rounding(at_precision(field, coeff_dtype), slices)
        key = str(coeff_dtype)
        if not rounding <= worst.get(key, {}).get("relative_rounding", -1.0):
            worst[key] = {
                "commit": commit,
                "coeff_dtype": key,
                "seed": seed,
                "slices": slices,
                "modes": (time_modes, space_modes),
                "field": name,
                "relative_rounding": rounding,
            }
for record in worst.values():
    json.dump(record, sys.stdout)
    sys.stdout.write("\n")
