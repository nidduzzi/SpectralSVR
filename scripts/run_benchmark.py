"""Regenerate the operator-baseline comparison table.

Fits every operator baseline on the same problems over a seed set and writes
one long CSV: one row per dataset, operator and seed, with the shared metric
suite (``utils.get_metrics``) and the provenance of the run on every row.

``just bench`` writes the record, ``benchmarks/operator_comparison.csv``, over
the default seed set. Any other run defaults to ``results/`` (gitignored), so a
check never overwrites the record. ``--append`` resumes an interrupted run: rows
already in the file are kept and not recomputed. ``operators.summarise`` reduces
the long table to a centre and a spread per dataset and operator.

Usage
-----
    uv run --extra baselines python scripts/run_benchmark.py --help
    ... --datasets antiderivative burgers-mms   # subset
    ... --operators spectralsvr                 # skip the neural baseline
    ... --quick                                 # small sizes, few iterations
    ... --device cuda                           # place every stage on an accelerator
    ... --seeds 0 1                             # a subset of the seed set
    ... --append                                # resume into an existing file

DeepONet needs the optional ``baselines`` extra (deepxde). Requesting it without
that extra fails rather than producing a table short of a row; pass
``--allow-missing-operators`` to benchmark without it deliberately.

What the table establishes
--------------------------
The harness holds the data and the metrics fixed: every operator is fitted on
the same sensor-sampled inputs and scored against the same targets at the same
query points through one metric suite. It does not hold the tuning fixed: the
hyperparameters on both sides are fixed defaults, chosen once and not tuned per
dataset. The reported figures come from the seed set 0, 1, 2, 3, 4: each row is
one seed, and a centre and spread over the set is what a reader should compare,
since one seed's figure varies with the draw. Every row records the seed set of
its run and whether it was a ``--quick`` check, so a row from a check is never
read as the record. The table demonstrates that the harness works; with the
tuning uncontrolled it is not a controlled comparison of the methods.

Run-to-run spread at one fixed seed, measured per operator: on cuda it is zero
for both SpectralSVR and DeepONet, since three runs of seed 0, one with the
datasets in reverse order, agreed bit for bit on every metric. On the host,
SpectralSVR's multi-threaded solve moved by up to 1.9e-05 relative between
runs; DeepONet does not run there. So a spread over the seed set measured on
cuda is the methods' variation across draws, not repetition noise.

A table is reproducible only against the same ``--device``. The device is
passed to every generator, every ``generate`` call and every query grid rather
than set as a process default, so no later import can move it -- an optional
baseline sets one on import. But a generator on another device draws different
numbers from the same seed, so ``--device cuda`` builds a different dataset
rather than reproducing the host table. Every operator fits on the stated
device too, and each row records both its data device and its fit device, which
agree. DeepONet trains only on an accelerator, so a table with DeepONet states
one (``just bench`` uses ``--device cuda``); the host is refused before anything
runs.

``fit_seconds`` is not like-for-like either. ``DeepONetOperator.fit`` calls
``_import_deepxde()`` -- a lazy import paid on its first call -- and converts the
dataset to numpy; ``SpectralSVROperator.fit`` does neither. Read the column as
the cost of each adapter as written, not of each method. Each row's
``fit_covers`` states the work its fit did -- one direct solve, or a fixed count
of training iterations -- so the times are read as two facts, not one ratio.

DeepONet is trained on each dataset by a named recipe taken from reference code,
with a training length chosen per dataset by a five-seed measurement: "Lu 2021 /
deepxde example" on the antiderivative rows and "Lu 2022 Burgers" on the Burgers
rows. Each row's ``protocol`` names the recipe, its source, and the training
length against the source's, which is shorter in every case.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch

from SpectralSVR import Antiderivative, Burgers, FourierBasis, LSSVR
from SpectralSVR.basis import Basis
from SpectralSVR.operators import (
    DeepONetOperator,
    DeepONetRecipe,
    Lu2021DeepxdeExample,
    Lu2022Burgers,
    Operator,
    Split,
    SpectralSVROperator,
    benchmark,
)
from SpectralSVR.utils import (
    DEFAULT_COEFF_DTYPE,
    etdrk4_solver,
    placed,
    resolve_device,
)

DEFAULT_OUT = Path("results") / "bench_results.csv"
RECORD_OUT = Path("benchmarks") / "operator_comparison.csv"
SEED_SET = (0, 1, 2, 3, 4)
ACCELERATOR_MEMORY_FRACTION = 0.25


class RunProvenance(NamedTuple):
    """What produced a run: code revision, whether the tree was clean, versions, precision."""

    revision: str
    tree_clean: bool
    torch_version: str
    deepxde_version: str
    coeff_dtype: str

    @classmethod
    def of_repository(cls, root: Path) -> RunProvenance:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        try:
            deepxde_version = importlib.metadata.version("deepxde")
        except importlib.metadata.PackageNotFoundError:
            deepxde_version = "absent"
        return cls(
            revision=git("rev-parse", "HEAD"),
            tree_clean=git("status", "--porcelain") == "",
            torch_version=torch.__version__,
            deepxde_version=deepxde_version,
            coeff_dtype=str(DEFAULT_COEFF_DTYPE),
        )


class BenchmarkCase(NamedTuple):
    """The split both operators see, and the points every prediction is scored at."""

    split: Split
    query: torch.Tensor


CaseBuilder = Callable[[int, bool, torch.device], BenchmarkCase]


class DeepONetPlan(NamedTuple):
    """The recipe and training length DeepONet is fitted with on one dataset."""

    recipe: DeepONetRecipe
    iterations: int


class DatasetSpec(NamedTuple):
    """A CLI dataset choice: its label in the CSV, how to build it, and DeepONet's plan for it."""

    label: str
    build: CaseBuilder
    deeponet: DeepONetPlan


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
    return FourierBasis(field.coeff[:, time_index], domain=space_domain)


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
    """f -> u(t, x) from rest, the forcing drawn first from a time-space GRF."""
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
    split = Split.partition(drawn_forcing, u, n_train, n_sensors=modes)
    query_times = 12
    slice_stride = (time_steps - 1) // (query_times - 1)
    time_axis = torch.arange(
        0, time_steps, slice_stride, dtype=torch.float64, device=device
    ) / (time_steps - 1)
    space_axis = torch.linspace(0, 1, 12, dtype=torch.float64, device=device)
    return BenchmarkCase(split, tensor_product_query(time_axis, space_axis))


DATASETS: dict[str, DatasetSpec] = {
    "antiderivative": DatasetSpec(
        "project antiderivative (ut->u)",
        antiderivative_case,
        DeepONetPlan(Lu2021DeepxdeExample(), 10_000),
    ),
    "burgers-mms": DatasetSpec(
        "project Burgers MMS (f->u)",
        burgers_mms_case,
        DeepONetPlan(Lu2022Burgers(periods=(None, 1.0)), 50_000),
    ),
    "burgers-numerical": DatasetSpec(
        "project Burgers numerical (u0 -> u(T))",
        burgers_numerical_case,
        DeepONetPlan(Lu2022Burgers(periods=(1.0,)), 10_000),
    ),
    "antiderivative-grf": DatasetSpec(
        "DeepONet paper antiderivative (GRF)",
        antiderivative_grf_case,
        DeepONetPlan(Lu2021DeepxdeExample(), 50_000),
    ),
    "burgers-forced": DatasetSpec(
        "project Burgers forced (f->u, GRF forcing, from rest)",
        burgers_forced_case,
        DeepONetPlan(Lu2022Burgers(periods=(None, 1.0)), 10_000),
    ),
}


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
class MissingOperatorDependency(RuntimeError):
    """A requested operator could not be built because an extra is not installed."""


OPERATOR_ROW_NAMES = {
    "spectralsvr": SpectralSVROperator.name,
    "deeponet": DeepONetOperator.name,
}


def build_operators(
    names: list[str],
    query: torch.Tensor,
    deeponet: DeepONetPlan,
    seed: int,
    device: torch.device,
    allow_missing: bool = False,
) -> list[Operator]:
    """Build every requested operator, or say which could not be built.

    A table with fewer rows than were asked for is reachable only through
    ``allow_missing``, so the default path cannot quietly drop a method.
    """
    operators: list[Operator] = []
    if "spectralsvr" in names:
        operators.append(
            SpectralSVROperator(
                FourierBasis(), LSSVR(kernel="rbf", C=50.0, device=device)
            )
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
                DeepONetOperator(
                    trunk=query,
                    recipe=deeponet.recipe,
                    device=device,
                    iterations=deeponet.iterations,
                    seed=seed,
                )
            )
    return operators


def main(argv: list[str] | None = None) -> None:
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
        "--append",
        action="store_true",
        help="keep the rows already in --out and add only the missing ones",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "device every stage runs on; stated rather than inherited, because "
            "importing an optional baseline sets the process default itself"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEED_SET))
    parser.add_argument(
        "--allow-missing-operators",
        action="store_true",
        help="benchmark without a requested operator whose extra is not installed",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="DeepONet training iterations on every dataset, overriding each plan",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small splits + 200 DeepONet iterations (smoke run)",
    )
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(ACCELERATOR_MEMORY_FRACTION)
    device = resolve_device(args.device)
    if "deeponet" in args.operators and not DeepONetOperator.trains_on(device):
        parser.error(
            f"DeepONet trains only on an accelerator; got --device {device}. "
            "State one (e.g. --device cuda) or drop it with --operators spectralsvr"
        )
    print(f"running on {device}")

    provenance = RunProvenance.of_repository(Path(__file__).resolve().parent)
    seed_set = str(sorted(args.seeds))
    iterations_override = 200 if args.quick else args.iterations
    recorded = (
        pd.read_csv(args.out)
        if args.append and args.out.exists()
        else pd.DataFrame(columns=["dataset", "operator", "seed"])
    )
    done = set(
        zip(recorded["dataset"], recorded["operator"], recorded["seed"], strict=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    recorded.to_csv(args.out, index=False)

    rows_added = 0
    for seed in args.seeds:
        for key in args.datasets:
            spec = DATASETS[key]
            missing = [
                choice
                for choice in args.operators
                if (spec.label, OPERATOR_ROW_NAMES[choice], seed) not in done
            ]
            if not missing:
                print(f"[{key} seed {seed}] already recorded")
                continue
            print(f"[{key} seed {seed}] building...")
            case = spec.build(seed, args.quick, device)
            plan = (
                spec.deeponet
                if iterations_override is None
                else spec.deeponet._replace(iterations=iterations_override)
            )
            operators = build_operators(
                missing,
                case.query,
                plan,
                seed,
                device,
                allow_missing=args.allow_missing_operators,
            )
            if not operators:
                print(f"[{key} seed {seed}] no operators available; skipped")
                continue
            table = benchmark(operators, case.split, case.query).reset_index()
            table.insert(0, "dataset", spec.label)
            table.insert(2, "seed", seed)
            table["data_device"] = str(placed(device))
            table["quick"] = args.quick
            table["seed_set"] = seed_set
            for field, value in provenance._asdict().items():
                table[field] = value
            recorded = pd.concat([recorded, table], ignore_index=True)
            recorded.to_csv(args.out, index=False)
            rows_added += len(table)

    if rows_added == 0 and recorded.empty:
        raise SystemExit("nothing benchmarked")
    print(f"\nwrote {args.out}")
    print(recorded.to_string(index=False))


if __name__ == "__main__":
    main()
