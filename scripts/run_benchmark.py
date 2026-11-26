"""Regenerate the operator-baseline comparison table.

Fits every operator baseline on the same problems and writes one CSV with a
``dataset`` column plus the shared metric suite (``utils.get_metrics``).

The output is generated data, not source: it lands in ``results/`` (gitignored)
so a rerun never dirties the working tree.

Usage
-----
    uv run --extra baselines python scripts/run_benchmark.py --help
    ... --datasets antiderivative burgers-mms   # subset
    ... --operators spectralsvr                 # skip the neural baseline
    ... --quick                                 # small sizes, few iterations
    ... --device cuda                           # place every stage on an accelerator

DeepONet needs the optional ``baselines`` extra (deepxde). Requesting it without
that extra fails rather than producing a table short of a row; pass
``--allow-missing-operators`` to benchmark without it deliberately.

What the table establishes
--------------------------
The harness holds the data and the metrics fixed: every operator is fitted on
the same sensor-sampled inputs and scored against the same targets at the same
query points through one metric suite. It does not hold the tuning fixed: the
hyperparameters on both sides are fixed defaults, chosen once and not tuned per
dataset, and each figure comes from a single seed (``--seed``, default 0). The table therefore demonstrates that the harness
works; it is not a controlled comparison of the methods.

A table is reproducible only against the same ``--device``. The device is
passed to every generator, every ``generate`` call and every query grid rather
than set as a process default, so no later import can move it -- an optional
baseline sets one on import. But a generator on another device draws different
numbers from the same seed, so ``--device cuda`` builds a different dataset
rather than reproducing the host table. Report the device alongside any figure.

``fit_seconds`` is not like-for-like either. ``DeepONetOperator.fit`` calls
``_import_deepxde()`` -- a lazy import paid on its first call -- and converts the
dataset to numpy; ``SpectralSVROperator.fit`` does neither. Read the column as
the cost of each adapter as written, not of each method.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch

from SpectralSVR import Antiderivative, Burgers, FourierBasis, LSSVR
from SpectralSVR.basis import Basis
from SpectralSVR.operators import (
    DeepONetOperator,
    Operator,
    Split,
    SpectralSVROperator,
    benchmark,
)
from SpectralSVR.utils import DEFAULT_COEFF_DTYPE, etdrk4_solver, resolve_device

DEFAULT_OUT = Path("results") / "bench_results.csv"


class BenchmarkCase(NamedTuple):
    """The split both operators see, and the points every prediction is scored at."""

    split: Split
    query: torch.Tensor


CaseBuilder = Callable[[int, bool, torch.device], BenchmarkCase]


class DatasetSpec(NamedTuple):
    """A CLI dataset choice: its label in the CSV, and how to build it."""

    label: str
    build: CaseBuilder


# --------------------------------------------------------------------------- #
# query grids
# --------------------------------------------------------------------------- #
def line_query(n: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0, 1, n, dtype=torch.float64, device=device).unsqueeze(-1)


def tensor_product_query(*axes: torch.Tensor) -> torch.Tensor:
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.flatten() for axis in mesh], dim=-1)


# --------------------------------------------------------------------------- #
# input-function sampling
# --------------------------------------------------------------------------- #
def signed_mode_numbers(size: int, device: torch.device) -> torch.Tensor:
    return torch.fft.fftfreq(size, device=device) * size


def grf_coeff(
    length_scale: float, domain_length: float = 1.0
) -> Callable[..., torch.Tensor]:
    """Coefficient sampler for a periodic squared-exponential GRF.

    A Gaussian random field with kernel ``exp(-|x - y|^2 / (2 l^2))`` has
    spectral density ``exp(-2 pi^2 l^2 xi^2)``; on a periodic domain of length
    ``L`` the mode-``m`` amplitude is therefore ``exp(-pi^2 l^2 m^2 / L^2)``.
    Drawing white noise and scaling by that amplitude samples the field
    directly in coefficient space.

    Plugged into :meth:`FourierBasis.generate` as ``random_func``, so the caller
    still gets Hermitian symmetry (real-valued functions) for free.

    Note this is the *periodic* analogue of the DeepONet paper's GRF input
    sampling, not a byte-exact replica of it -- the paper samples on a
    non-periodic interval, which a Fourier basis cannot represent.
    """

    def sample(
        size: tuple[int, ...],
        dtype: torch.dtype = DEFAULT_COEFF_DTYPE,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        white = torch.randn(size, dtype=dtype, generator=generator, device=device)
        for axis, n_modes in enumerate(size[1:], start=1):
            modes = signed_mode_numbers(n_modes, white.device)
            amplitude = torch.exp(
                -(math.pi**2) * length_scale**2 * modes.square() / domain_length**2
            )
            broadcast = [-1 if dim == axis else 1 for dim in range(white.ndim)]
            white = white * amplitude.reshape(broadcast)
        return white

    return sample


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def spatial_field_at(field: Basis, time_index: int) -> FourierBasis:
    """One time slice of a time-dependent field, as a field over space alone."""
    assert field.coeff is not None
    space_domain = (field.domain[1],)
    return FourierBasis(field.coeff[:, time_index].cpu(), domain=space_domain)


def antiderivative_case(
    seed: int, quick: bool, device: torch.device, random_func=torch.randn
) -> BenchmarkCase:
    """ut -> u: integrate a band-limited derivative back to its antiderivative."""
    n_train, n_test = (40, 10) if quick else (120, 30)
    modes = 16
    generator = torch.Generator(device=device).manual_seed(seed)
    u, ut = Antiderivative().generate(
        FourierBasis,
        n_train + n_test,
        modes,
        generator=generator,
        u0=0,
        random_func=random_func,
        device=device,
    )
    split = Split.partition(ut, u, n_train, n_sensors=modes)
    return BenchmarkCase(split, line_query(40, device))


def antiderivative_grf_case(
    seed: int, quick: bool, device: torch.device
) -> BenchmarkCase:
    """Same operator as :func:`antiderivative_case`, GRF-sampled inputs."""
    return antiderivative_case(
        seed, quick, device, random_func=grf_coeff(length_scale=0.2)
    )


def burgers_mms_case(seed: int, quick: bool, device: torch.device) -> BenchmarkCase:
    """f -> u on manufactured viscous-Burgers solutions, over time and space."""
    n_train, n_test = (24, 6) if quick else (60, 15)
    modes = 8 if quick else 12
    generator = torch.Generator(device=device).manual_seed(seed)
    u, f = Burgers().generate(
        FourierBasis,
        n_train + n_test,
        modes,
        generator=generator,
        u0="random",
        f="random",
        nu=0.01,
        time_dependent_coeff=False,
        device=device,
    )
    split = Split.partition(f, u, n_train, n_sensors=modes)
    time_axis = torch.linspace(0, 1, 12, dtype=torch.float64, device=device)
    space_axis = torch.linspace(0, 1, 12, dtype=torch.float64, device=device)
    return BenchmarkCase(split, tensor_product_query(time_axis, space_axis))


def burgers_numerical_case(
    seed: int, quick: bool, device: torch.device
) -> BenchmarkCase:
    """u0 -> u(T): the ETDRK4-simulated final state from the initial condition."""
    n_train, n_test = (16, 4) if quick else (40, 10)
    modes = 16
    n_steps = 40 if quick else 100
    generator = torch.Generator(device=device).manual_seed(seed)
    u, _ = Burgers().generate(
        FourierBasis,
        n_train + n_test,
        modes,
        generator=generator,
        u0="random",
        f=0,
        nu=0.01,
        time_domain=slice(0, 1, n_steps),
        solver=etdrk4_solver,
        device=device,
    )
    initial = spatial_field_at(u, 0)
    final = spatial_field_at(u, -1)
    split = Split.partition(initial, final, n_train, n_sensors=modes)
    return BenchmarkCase(split, line_query(40, device))


FORCING_LENGTH_SCALE = 0.2
FORCING_SAMPLER = grf_coeff(FORCING_LENGTH_SCALE)


def periodic_in_time(u: FourierBasis) -> FourierBasis:
    """``u`` over ``(time, space)``, periodic in time and exact at its slices.

    ``u``'s slices span its time interval inclusively, so every slice but the
    last lies on the periodic grid of that interval. The last, at the
    interval's end, is not represented.
    """
    periodic_slices = u.time_size - 1
    slice_values = FourierBasis.inv_transform(
        u.coeff[:, :periodic_slices].flatten(0, 1)
    ).reshape(len(u), periodic_slices, *u.coeff.shape[2:])
    return FourierBasis(
        coeff=FourierBasis.transform(slice_values),
        domain=u.domain,
        time_dependent=False,
    )


def burgers_forced_case(
    seed: int,
    quick: bool,
    device: torch.device,
    *,
    n_train: int | None = None,
    n_test: int | None = None,
    modes: int | None = None,
    random_func: Callable[..., torch.Tensor] = FORCING_SAMPLER,
) -> BenchmarkCase:
    """f -> u from rest, the forcing drawn first from a time-space GRF.

    Pilot: ``u`` is scored through :func:`periodic_in_time`, only at grid
    times, so ``u(T)`` is not scored.
    """
    quick_or_full = (24, 6, 8, 45) if quick else (60, 15, 12, 89)
    default_train, default_test, default_modes, time_steps = quick_or_full
    n_train = default_train if n_train is None else n_train
    n_test = default_test if n_test is None else n_test
    modes = default_modes if modes is None else modes
    generator = torch.Generator(device=device).manual_seed(seed)
    drawn_forcing = FourierBasis.generate(
        n_train + n_test,
        (modes, modes),
        generator=generator,
        random_func=random_func,
        domain=((0.0, 1.0), (0.0, 1.0)),
        device=device,
    )
    u, _ = Burgers().generate(
        FourierBasis,
        n_train + n_test,
        modes,
        u0=0,
        f=drawn_forcing,
        nu=0.01,
        time_domain=slice(0, 1, time_steps),
        solver=etdrk4_solver,
        device=device,
    )
    split = Split.partition(
        drawn_forcing, periodic_in_time(u), n_train, n_sensors=modes
    )
    periodic_slices = time_steps - 1
    query_times = 11
    time_axis = (
        torch.arange(
            0,
            periodic_slices,
            periodic_slices // query_times,
            dtype=torch.float64,
            device=device,
        )
        / periodic_slices
    )
    space_axis = torch.linspace(0, 1, 12, dtype=torch.float64, device=device)
    return BenchmarkCase(split, tensor_product_query(time_axis, space_axis))


DATASETS: dict[str, DatasetSpec] = {
    "antiderivative": DatasetSpec(
        "project antiderivative (ut->u)", antiderivative_case
    ),
    "burgers-mms": DatasetSpec("project Burgers MMS (f->u)", burgers_mms_case),
    "burgers-numerical": DatasetSpec(
        "project Burgers numerical (u0 -> u(T))", burgers_numerical_case
    ),
    "antiderivative-grf": DatasetSpec(
        "DeepONet paper antiderivative (GRF)", antiderivative_grf_case
    ),
    "burgers-forced": DatasetSpec(
        "project Burgers forced (f->u, GRF forcing, from rest, pilot: t periodic)",
        burgers_forced_case,
    ),
}


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
class MissingOperatorDependency(RuntimeError):
    """A requested operator could not be built because an extra is not installed."""


def build_operators(
    names: list[str],
    query: torch.Tensor,
    iterations: int,
    seed: int,
    allow_missing: bool = False,
) -> list[Operator]:
    """Build every requested operator, or say which could not be built.

    A table with fewer rows than were asked for is reachable only through
    ``allow_missing``, so the default path cannot quietly drop a method.
    """
    operators: list[Operator] = []
    if "spectralsvr" in names:
        operators.append(
            SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0))
        )
    if "deeponet" in names:
        if importlib.util.find_spec("deepxde") is None:
            if not allow_missing:
                raise MissingOperatorDependency(
                    "DeepONet was requested but deepxde is not installed "
                    "(uv sync --extra baselines). Pass --allow-missing-operators "
                    "to benchmark without it, accepting a table that omits it."
                )
            print("  ! deepxde missing; omitting DeepONet (--allow-missing-operators)")
        else:
            operators.append(
                DeepONetOperator(trunk=query, iterations=iterations, seed=seed)
            )
    return operators


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=["spectralsvr", "deeponet"],
        default=["spectralsvr", "deeponet"],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "device every stage runs on; stated rather than inherited, because "
            "importing an optional baseline sets the process default itself"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-missing-operators",
        action="store_true",
        help="benchmark without a requested operator whose extra is not installed",
    )
    parser.add_argument(
        "--iterations", type=int, default=10_000, help="DeepONet training iterations"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small splits + 200 DeepONet iterations (smoke run)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"running on {device}")

    iterations = 200 if args.quick else args.iterations
    frames = []
    for key in args.datasets:
        spec = DATASETS[key]
        print(f"[{key}] building...")
        case = spec.build(args.seed, args.quick, device)
        operators = build_operators(
            args.operators,
            case.query,
            iterations,
            args.seed,
            allow_missing=args.allow_missing_operators,
        )
        if not operators:
            print(f"[{key}] no operators available; skipped")
            continue
        print(
            f"[{key}] fitting {len(operators)} operator(s)"
            f" on {case.split.train.n_functions} functions"
        )
        table = benchmark(operators, case.split, case.query).reset_index()
        table.insert(0, "dataset", spec.label)
        frames.append(table)

    if not frames:
        raise SystemExit("nothing benchmarked")

    result = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
