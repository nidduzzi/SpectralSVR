import itertools

import torch

from SpectralSVR import FourierBasis
from SpectralSVR.basis.strategy import EvaluationStrategy

EXACT = EvaluationStrategy(allow_approximate=False)
APPROXIMATE = EvaluationStrategy(memory_budget_mb=0.0, allow_approximate=True)
SEEDS = range(5)
POINTS = 200
RANGES = {1: range(1, 49), 2: range(1, 13), 3: range(1, 13)}


def relative_error(modes: tuple[int, ...], seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    coeff = FourierBasis.generate_coeff(2, modes, generator=generator).to(
        torch.complex128
    )
    points = torch.rand(POINTS, len(modes), generator=generator, dtype=torch.float64)
    domain = tuple((0.0, 1.0) for _ in modes)
    exact = FourierBasis.evaluate(coeff, points, domain=domain, strategy=EXACT)
    approximate = FourierBasis.evaluate(
        coeff, points, domain=domain, strategy=APPROXIMATE
    )
    return float((approximate - exact).abs().max() / exact.abs().max())


if __name__ == "__main__":
    for ndim, counts in RANGES.items():
        worst, worst_small = 0.0, 0.0
        for modes in itertools.product(counts, repeat=ndim):
            error = max(relative_error(modes, seed) for seed in SEEDS)
            worst = max(worst, error)
            if min(modes) <= 2:
                worst_small = max(worst_small, error)
        print(
            f"{ndim} axes, modes {counts.start} to {counts.stop - 1}: worst {worst:.2e}, "
            f"worst with a 1- or 2-mode axis {worst_small:.2e}"
        )
