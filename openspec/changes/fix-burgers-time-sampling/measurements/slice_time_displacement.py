import itertools
import json
import math
import subprocess
import sys

import torch

from SpectralSVR import FourierBasis
from SpectralSVR.problems import Burgers

NU = 0.05
UNIT = (0.0, 1.0)
OFFSET_TIME = (0.5, 3.2)
OFFSET_SPACE = (-1.0, 2.0)
CASES = (
    *(
        dict(n=2, slices=slices, time_modes=8, space_modes=8, time_domain=UNIT, space_domain=UNIT, seed=0)
        for slices in (2, 5, 11, 50)
    ),
    dict(n=1, slices=2, time_modes=8, space_modes=8, time_domain=UNIT, space_domain=UNIT, seed=0),
    dict(n=2, slices=8, time_modes=8, space_modes=8, time_domain=UNIT, space_domain=UNIT, seed=0),
    dict(n=2, slices=7, time_modes=7, space_modes=9, time_domain=UNIT, space_domain=UNIT, seed=0),
    dict(n=2, slices=4, time_modes=9, space_modes=7, time_domain=UNIT, space_domain=UNIT, seed=0),
    dict(n=3, slices=12, time_modes=9, space_modes=6, time_domain=UNIT, space_domain=UNIT, seed=0),
    dict(n=2, slices=9, time_modes=6, space_modes=13, time_domain=OFFSET_TIME, space_domain=OFFSET_SPACE, seed=0),
)
SWEEP_SLICES = range(2, 41)
SWEEP_MODES = range(6, 17)
SWEEP_STARTS = (-3.0, 3.0)
SWEEP_LENGTHS = (0.5, 3.0)


def clean_checkout_commit():
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout
    if changed:
        raise SystemExit(f"tracked files differ from the checked-out commit:\n{changed}")
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def fourier_series_at(coeff, domain, times, space):
    (t_start, t_stop), (x_start, x_stop) = domain
    time_modes, space_modes = coeff.shape[-2:]
    time_wavenumbers = torch.fft.fftfreq(time_modes, 1 / time_modes, dtype=torch.float64)
    space_wavenumbers = torch.fft.fftfreq(space_modes, 1 / space_modes, dtype=torch.float64)
    time_phase = torch.exp(2j * torch.pi * torch.outer(times - t_start, time_wavenumbers) / (t_stop - t_start))
    space_phase = torch.exp(2j * torch.pi * torch.outer(space - x_start, space_wavenumbers) / (x_stop - x_start))
    series = torch.einsum("nkl,tk,xl->ntx", coeff.to(torch.complex128), time_phase, space_phase)
    return series / (time_modes * space_modes)


def relative_displacements(n, slices, time_modes, space_modes, time_domain, space_domain, seed):
    def manufacture(time_dependent_coeff):
        return Burgers().generate(
            FourierBasis,
            n,
            (time_modes, space_modes),
            u0="random",
            f="random",
            nu=NU,
            time_domain=slice(*time_domain, slices),
            space_domain=slice(*space_domain, space_modes),
            time_dependent_coeff=time_dependent_coeff,
            generator=torch.Generator().manual_seed(seed),
        )

    times = torch.linspace(*time_domain, slices, dtype=torch.float64)
    space = torch.linspace(*space_domain, space_modes + 1, dtype=torch.float64)[:-1]
    displacements = {}
    for name, stored, manufactured in zip(("u", "f"), manufacture(True), manufacture(False), strict=True):
        expected = fourier_series_at(manufactured.coeff, manufactured.domain, times, space)
        relative = float((stored.get_values() - expected).abs().max() / expected.abs().max())
        if not math.isfinite(relative):
            raise SystemExit(f"non-finite displacement of {name}: slices={slices}, modes=({time_modes}, {space_modes}), seed={seed}")
        displacements[name] = relative
    return displacements


def swept_domain(draw):
    starts, lengths = torch.rand(2, 2, generator=torch.Generator().manual_seed(draw), dtype=torch.float64)
    time_start, space_start = (SWEEP_STARTS[0] + starts * (SWEEP_STARTS[1] - SWEEP_STARTS[0])).tolist()
    time_length, space_length = (SWEEP_LENGTHS[0] + lengths * (SWEEP_LENGTHS[1] - SWEEP_LENGTHS[0])).tolist()
    return (time_start, time_start + time_length), (space_start, space_start + space_length)


def record_cases(commit):
    for case in CASES:
        json.dump({"commit": commit, "record": "case", **case, **relative_displacements(**case)}, sys.stdout)
        sys.stdout.write("\n")


def record_sweep_worst(commit):
    worst = {"u": dict(relative=-1.0), "f": dict(relative=-1.0)}
    draws = list(itertools.product(SWEEP_SLICES, SWEEP_MODES, SWEEP_MODES))
    for draw, (slices, time_modes, space_modes) in enumerate(draws):
        time_domain, space_domain = swept_domain(draw)
        case = dict(
            n=2,
            slices=slices,
            time_modes=time_modes,
            space_modes=space_modes,
            time_domain=time_domain,
            space_domain=space_domain,
            seed=draw,
        )
        for name, relative in relative_displacements(**case).items():
            if relative > worst[name]["relative"]:
                worst[name] = dict(relative=relative, **case)
    for name, record in worst.items():
        json.dump({"commit": commit, "record": "sweep_worst", "draws": len(draws), "field": name, **record}, sys.stdout)
        sys.stdout.write("\n")


RECORDS = {"cases": record_cases, "sweep": record_sweep_worst}
RECORDS[sys.argv[1]](clean_checkout_commit())
