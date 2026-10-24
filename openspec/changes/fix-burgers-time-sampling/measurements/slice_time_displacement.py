"""Largest distance between each stored slice of a time-dependent manufactured pair and the manufactured field at that slice's inclusive time."""

import json
import subprocess
import sys

import torch

from SpectralSVR import FourierBasis
from SpectralSVR.problems import Burgers

SEED = 0
FUNCTIONS = 2
MODES = (8, 8)
NU = 0.05
UNIT = (0.0, 1.0)
SLICE_COUNTS = (2, 5, 11, 50)


def manufactured_pair(slices, time_dependent_coeff):
    return Burgers().generate(
        FourierBasis,
        FUNCTIONS,
        MODES,
        u0="random",
        f="random",
        nu=NU,
        space_domain=slice(*UNIT, 200),
        time_domain=slice(*UNIT, slices),
        time_dependent_coeff=time_dependent_coeff,
        generator=torch.Generator().manual_seed(SEED),
    )


def slice_points(slices, space_points):
    times = torch.linspace(*UNIT, slices, dtype=torch.float64)
    space = UNIT[0] + torch.arange(space_points, dtype=torch.float64) * (UNIT[1] - UNIT[0]) / space_points
    mesh = torch.meshgrid(times, space, indexing="ij")
    return torch.stack([axis.flatten() for axis in mesh], dim=-1)


def relative_displacement(stored, manufactured, slices):
    expected = manufactured(slice_points(slices, MODES[1]))
    return float((stored.get_values().flatten(1) - expected).abs().max() / expected.abs().max())


commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
for slices in SLICE_COUNTS:
    stored = manufactured_pair(slices, time_dependent_coeff=True)
    manufactured = manufactured_pair(slices, time_dependent_coeff=False)
    for name, stored_field, manufactured_field in zip(("u", "f"), stored, manufactured, strict=True):
        json.dump(
            {
                "commit": commit,
                "seed": SEED,
                "modes": MODES,
                "slices": slices,
                "field": name,
                "relative_displacement": relative_displacement(stored_field, manufactured_field, slices),
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
