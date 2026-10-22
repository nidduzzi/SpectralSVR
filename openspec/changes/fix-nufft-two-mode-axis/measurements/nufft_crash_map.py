import itertools

import torch

from SpectralSVR.basis._nufft import nufft_evaluate

MODES_PER_AXIS = range(1, 7)
AXIS_COUNTS = (1, 2, 3)


def failing_mode_tuples() -> set[tuple[int, ...]]:
    generator = torch.Generator().manual_seed(0)
    failing = set()
    for ndim in AXIS_COUNTS:
        for modes in itertools.product(MODES_PER_AXIS, repeat=ndim):
            coeff = torch.randn(2, *modes, dtype=torch.complex128, generator=generator)
            points = torch.rand(5, ndim, dtype=torch.float64, generator=generator)
            try:
                nufft_evaluate(coeff, points, tuple((0.0, 1.0) for _ in modes))
            except IndexError:
                failing.add(modes)
    return failing


if __name__ == "__main__":
    failing = failing_mode_tuples()
    with_small_axis = {
        modes
        for ndim in AXIS_COUNTS
        for modes in itertools.product(MODES_PER_AXIS, repeat=ndim)
        if min(modes) <= 2
    }
    print(f"{len(failing)} failing mode tuples")
    print(
        f"exactly the tuples with an axis of 1 or 2 modes: {failing == with_small_axis}"
    )
