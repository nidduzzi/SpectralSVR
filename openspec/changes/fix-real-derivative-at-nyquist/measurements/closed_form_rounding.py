import json
import subprocess
import sys
from pathlib import Path

import numpy
import torch

sys.path.insert(0, str(Path.cwd()))
from tests._anchors import fourier_series_at  # noqa: E402
from tests._time_varying import grid_points, rows  # noqa: E402

from SpectralSVR import FourierBasis  # noqa: E402
from SpectralSVR.basis.strategy import EvaluationStrategy  # noqa: E402

DRAWS = 3000
EXACT = EvaluationStrategy(allow_approximate=False)
PRECISIONS = (torch.complex64, torch.complex128)


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


def drawn_field(draw, coeff_dtype):
    generator = torch.Generator().manual_seed(draw)
    axes = 1 + draw % 2
    modes = tuple(int(m) for m in torch.randint(2, 13, (axes,), generator=generator))
    complex_funcs = bool(draw % 3 == 0)
    starts = (
        torch.rand(axes, generator=generator, dtype=torch.float64) * 4 - 2
    ).tolist()
    lengths = (
        torch.rand(axes, generator=generator, dtype=torch.float64) * 2.5 + 0.5
    ).tolist()
    domain = tuple(
        (start, start + length) for start, length in zip(starts, lengths, strict=True)
    )
    coeff = FourierBasis.generate(
        1 + draw % 3, modes, complex_funcs=complex_funcs, generator=generator
    ).coeff.to(coeff_dtype)
    field = FourierBasis(
        coeff,
        domain=domain,
        complex_funcs=complex_funcs,
        coeff_dtype=coeff_dtype,
        strategy=EXACT,
    )
    points = torch.stack(
        [
            start + length * torch.rand(16, generator=generator, dtype=torch.float64)
            for start, length in zip(starts, lengths, strict=True)
        ],
        dim=1,
    )
    orders = tuple(int(o) for o in torch.randint(0, 4, (axes,), generator=generator))
    return field, points, orders


def series_bound(field, orders):
    bound = field.coeff.abs().to(torch.float64)
    for axis, (m, (start, stop), order) in enumerate(
        zip(field.modes, field.domain, orders, strict=True)
    ):
        eigenvalue = (
            2 * numpy.pi * numpy.abs(numpy.fft.fftfreq(m, 1 / m)) / (stop - start)
        ) ** order
        shape = [1] * bound.ndim
        shape[axis + 1] = m
        bound = bound * torch.from_numpy(eigenvalue).view(shape)
    return float(bound.flatten(1).sum(1).max()) / field.coeff[0].numel()


def relative(got, expected, scale):
    value = float(numpy.abs(got - expected).max() / scale)
    if not numpy.isfinite(value):
        raise SystemExit(f"non-finite comparison: scale {scale}")
    return value


def nodes(field):
    return (
        rows(
            *(
                grid_points(m, "cpu", interval)
                for m, interval in zip(field.modes, field.domain, strict=True)
            )
        )
        if field.ndim == 2
        else grid_points(field.modes[0], "cpu", field.domain[0]).view(-1, 1)
    )


def comparisons(field, points, orders):
    real = not field.complex_funcs
    coeff = field.coeff.to(torch.complex128)
    differentiated = field
    for axis, order in enumerate(orders):
        if order:
            differentiated = differentiated.grad(dim=axis, ord=order)
    on_nodes = nodes(field)
    yield (
        "reading off the grid",
        relative(
            field(points).numpy(),
            fourier_series_at(coeff, points, field.domain, real=real),
            series_bound(field, (0,) * field.ndim),
        ),
    )
    yield (
        "derivative on the grid",
        relative(
            differentiated.get_values().flatten(1).numpy(),
            fourier_series_at(
                coeff, on_nodes, field.domain, real=real, derivative_orders=orders
            ),
            series_bound(field, orders),
        )
        if any(orders)
        else None,
    )
    if real and field.modes[0] > 2:
        without_mean_or_boundary = field.coeff.clone()
        without_mean_or_boundary.select(1, 0).zero_()
        without_mean_or_boundary.select(1, field.modes[0] // 2).zero_()
        returned = (
            FourierBasis(
                without_mean_or_boundary,
                domain=field.domain,
                coeff_dtype=field.coeff.dtype,
            )
            .grad(dim=0)
            .integral(dim=0)
        )
        yield (
            "integral of the derivative",
            relative(
                returned.coeff.numpy(),
                without_mean_or_boundary.numpy(),
                float(without_mean_or_boundary.abs().max()),
            ),
        )
    if real:
        values = field(points)
        yield (
            "imaginary part off the grid",
            float(values.imag.abs().max()) / series_bound(field, (0,) * field.ndim),
        )


commit = clean_checkout_commit()
worst = {}
for coeff_dtype in PRECISIONS:
    for draw in range(DRAWS):
        field, points, orders = drawn_field(draw, coeff_dtype)
        for name, value in comparisons(field, points, orders):
            if value is None:
                continue
            key = (str(coeff_dtype), name)
            if not value <= worst.get(key, {}).get("worst", -1.0):
                worst[key] = {
                    "worst": value,
                    "draw": draw,
                    "modes": field.modes,
                    "complex_funcs": field.complex_funcs,
                    "orders": orders,
                }
for (coeff_dtype, name), record in sorted(worst.items()):
    json.dump(
        {
            "commit": commit,
            "coeff_dtype": coeff_dtype,
            "comparison": name,
            "draws": DRAWS,
            **record,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
