import importlib.util
import math
import pathlib

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

from SpectralSVR import accelerator
from SpectralSVR.utils import DEFAULT_COEFF_DTYPE

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
_spec = importlib.util.spec_from_file_location("_run_benchmark_forced", SCRIPT)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

HOST = torch.device("cpu")
DEVICES_PRESENT = [HOST] + ([accelerator()] if torch.accelerator.is_available() else [])
SEEDS = st.integers(0, 2**32 - 1)
SIZES = st.tuples(st.integers(1, 6), st.integers(1, 4), st.integers(2, 10))
SMALLEST_SIZES = (1, 1, 2)
QUICK_SIZES = (24, 6, 8)
FULL_SIZES = (60, 15, 12)
AT_REST_ATOL = 1e-6
PERIODIC_IN_TIME_RTOL = 2e-5
FUZZ = settings(deadline=None, max_examples=6)

needs_an_accelerator = pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="placement against a process default pointing elsewhere needs a second device",
)


def build_forced(seed, quick, device, sizes, **options):
    n_train, n_test, modes = sizes
    return bench.burgers_forced_case(
        seed, quick, device, n_train=n_train, n_test=n_test, modes=modes, **options
    )


def no_forcing(size, dtype=DEFAULT_COEFF_DTYPE, generator=None, device=None):
    return torch.zeros(size, dtype=dtype, device=device)


def the_same_case(first, second) -> bool:
    halves = zip(
        (first.split.train, first.split.test),
        (second.split.train, second.split.test),
        strict=True,
    )
    return torch.equal(first.query.cpu(), second.query.cpu()) and all(
        torch.equal(a.f.cpu(), b.f.cpu())
        and torch.equal(a.sensors.cpu(), b.sensors.cpu())
        and torch.equal(a.u.coeff.cpu(), b.u.coeff.cpu())
        for a, b in halves
    )


class BenchmarkCasesStayPut(RuleBasedStateMachine):
    built = Bundle("built")

    def __init__(self):
        super().__init__()
        self.first_builds = {}

    @rule(
        target=built,
        key=st.sampled_from(sorted(bench.DATASETS)),
        seed=st.integers(0, 3),
        device=st.sampled_from(DEVICES_PRESENT),
    )
    def build(self, key, seed, device):
        case = bench.DATASETS[key].build(seed, True, device)
        first = self.first_builds.setdefault((key, seed, device), case)
        assert the_same_case(case, first)
        return (key, seed, device)

    @rule(identity=built)
    def rebuild(self, identity):
        key, seed, device = identity
        case = bench.DATASETS[key].build(seed, True, device)
        assert the_same_case(case, self.first_builds[identity])

    @rule(seed=SEEDS, sizes=SIZES, device=st.sampled_from(DEVICES_PRESENT))
    def build_the_forced_case_at_other_sizes(self, seed, sizes, device):
        build_forced(seed, True, device, sizes)


BenchmarkCasesStayPut.TestCase.settings = settings(
    deadline=None, max_examples=10, stateful_step_count=10
)
TestBenchmarkCasesStayPut = pytest.mark.scenario(
    "Adding a row leaves the others alone"
)(pytest.mark.oracle_invariant(BenchmarkCasesStayPut.TestCase))


@pytest.mark.oracle_analytic
@FUZZ
@given(seed=SEEDS, sizes=SIZES, device=st.sampled_from(DEVICES_PRESENT))
@example(seed=0, sizes=SMALLEST_SIZES, device=HOST)
@example(seed=2**32 - 1, sizes=QUICK_SIZES, device=HOST)
def test_from_rest_the_solution_starts_at_zero(seed, sizes, device):
    case = build_forced(seed, True, device, sizes)
    at_the_start = case.query[case.query[:, 0] == 0]

    for half in (case.split.train, case.split.test):
        assert half.targets(at_the_start).abs().max() <= AT_REST_ATOL


@pytest.mark.oracle_analytic
@FUZZ
@given(quick=st.booleans(), sizes=SIZES, device=st.sampled_from(DEVICES_PRESENT))
@example(quick=False, sizes=FULL_SIZES, device=HOST)
@example(quick=True, sizes=SMALLEST_SIZES, device=HOST)
def test_rest_under_no_forcing_stays_at_rest(quick, sizes, device):
    case = build_forced(0, quick, device, sizes, random_func=no_forcing)

    for half in (case.split.train, case.split.test):
        assert torch.count_nonzero(half.u.coeff) == 0
        assert torch.count_nonzero(half.targets(case.query)) == 0


@pytest.mark.oracle_smoke
@FUZZ
@given(
    seed=SEEDS,
    quick=st.booleans(),
    sizes=SIZES,
    device=st.sampled_from(DEVICES_PRESENT),
)
@example(seed=0, quick=True, sizes=SMALLEST_SIZES, device=HOST)
@example(seed=2**32 - 1, quick=True, sizes=QUICK_SIZES, device=HOST)
@example(seed=0, quick=False, sizes=FULL_SIZES, device=HOST)
def test_the_forced_case_has_the_shape_the_harness_expects(seed, quick, sizes, device):
    n_train, n_test, modes = sizes
    case = build_forced(seed, quick, device, sizes)
    train, test = case.split.train, case.split.test
    sensors = modes**2

    assert train.f.shape == (n_train, sensors)
    assert test.f.shape == (n_test, sensors)
    assert train.sensors.shape == test.sensors.shape == (sensors, 2)
    assert case.query.shape[1] == 2
    for half in (train, test):
        assert torch.isfinite(half.f).all()
        assert torch.isfinite(half.u.coeff).all()
        assert torch.isfinite(half.targets(case.query)).all()


@pytest.mark.oracle_consistency
@FUZZ
@given(
    seed=SEEDS,
    other=SEEDS,
    sizes=SIZES,
    device=st.sampled_from(DEVICES_PRESENT),
)
@example(seed=0, other=1, sizes=SMALLEST_SIZES, device=HOST)
@example(seed=2**32 - 1, other=0, sizes=QUICK_SIZES, device=HOST)
def test_a_seed_fixes_the_case_and_another_seed_changes_it(seed, other, sizes, device):
    first = build_forced(seed, True, device, sizes)

    assert the_same_case(first, build_forced(seed, True, device, sizes))
    if other != seed:
        assert not torch.equal(
            first.split.train.f, build_forced(other, True, device, sizes).split.train.f
        )


@needs_an_accelerator
@pytest.mark.oracle_smoke
@FUZZ
@given(
    seed=SEEDS,
    sizes=SIZES,
    stated_on_the_accelerator=st.booleans(),
)
@example(seed=0, sizes=SMALLEST_SIZES, stated_on_the_accelerator=True)
@example(seed=0, sizes=SMALLEST_SIZES, stated_on_the_accelerator=False)
def test_the_stages_the_builder_places_follow_the_stated_device(
    seed, sizes, stated_on_the_accelerator
):
    stated, elsewhere = (
        (accelerator(), HOST) if stated_on_the_accelerator else (HOST, accelerator())
    )
    torch.set_default_device(elsewhere)

    case = build_forced(seed, True, stated, sizes)

    assert case.query.device.type == stated.type
    for half in (case.split.train, case.split.test):
        assert half.u.coeff.device.type == stated.type


@pytest.mark.oracle_invariant
@FUZZ
@given(
    seed=st.integers(0, 10_000),
    functions=st.integers(1, 4),
    slices=st.integers(2, 29),
    modes=st.integers(2, 15),
    device=st.sampled_from(DEVICES_PRESENT),
)
@example(seed=0, functions=1, slices=2, modes=2, device=HOST)
@example(seed=0, functions=4, slices=29, modes=15, device=HOST)
def test_periodic_in_time_passes_through_every_slice_but_the_last(
    seed, functions, slices, modes, device
):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(functions, slices, modes, generator=generator).to(device)
    per_slice = bench.FourierBasis(
        coeff=bench.FourierBasis.transform(values.flatten(0, 1)).reshape(
            functions, slices, modes
        ),
        domain=((0.0, 1.0), (0.0, 1.0)),
        time_dependent=True,
    )
    times = torch.arange(slices - 1, dtype=torch.float64, device=device) / (slices - 1)
    grid_points = torch.arange(modes, dtype=torch.float64, device=device) / modes
    end_of_space = torch.ones(1, dtype=torch.float64, device=device)

    periodic = bench.periodic_in_time(per_slice)
    on_the_grid = periodic(bench.tensor_product_query(times, grid_points)).real
    at_the_end_of_space = periodic(bench.tensor_product_query(times, end_of_space)).real

    tolerance = PERIODIC_IN_TIME_RTOL * values.abs().max()
    expected = values[:, : slices - 1].reshape(functions, -1)
    assert (on_the_grid - expected).abs().max() <= tolerance
    wrapped = on_the_grid.reshape(functions, slices - 1, modes)[:, :, 0]
    assert (at_the_end_of_space - wrapped).abs().max() <= tolerance


@pytest.mark.oracle_smoke
@pytest.mark.scenario("Forced dataset builds and scores")
@settings(deadline=None, max_examples=8)
@given(seed=SEEDS)
@example(seed=0)
@example(seed=2**32 - 1)
def test_the_forced_case_is_scored_with_finite_metrics(seed):
    case = bench.burgers_forced_case(seed, True, HOST)
    operators = bench.build_operators(
        ["spectralsvr"], case.query, iterations=0, seed=seed
    )

    table = bench.benchmark(operators, case.split, case.query)

    assert len(table) == 1
    metrics = table.drop(columns=["pred_nan_sum"]).select_dtypes("number")
    assert all(math.isfinite(value) for value in metrics.to_numpy().flatten())
    assert int(table["pred_nan_sum"].iloc[0]) == 0


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("Forced dataset is registered")
def test_every_registered_dataset_is_named_once():
    labels = [spec.label for spec in bench.DATASETS.values()]

    assert "burgers-forced" in bench.DATASETS
    assert len(set(labels)) == len(labels)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("The derived and drawn forcings are distinguishable")
def test_every_forcing_to_solution_row_says_how_its_forcing_was_obtained():
    provenance = {"MMS": "derived from a drawn solution", "GRF forcing": "drawn"}
    forcing_to_solution = [
        spec.label for spec in bench.DATASETS.values() if "(f->u" in spec.label
    ]

    stated = [
        [how for marker, how in provenance.items() if marker in label]
        for label in forcing_to_solution
    ]
    assert all(len(ways) == 1 for ways in stated)
    assert {ways[0] for ways in stated} == set(provenance.values())
