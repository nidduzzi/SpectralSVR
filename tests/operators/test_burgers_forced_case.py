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
END_OF_INTERVAL_RTOL = 5e-6
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
    tensor_pairs = [(first.query, second.query)] + [
        pair
        for a, b in halves
        for pair in ((a.f, b.f), (a.sensors, b.sensors), (a.u.coeff, b.u.coeff))
    ]
    return all(x.device == y.device and torch.equal(x, y) for x, y in tensor_pairs)


@pytest.mark.oracle_consistency
@pytest.mark.scenario("A slice keeps its field's device")
@pytest.mark.parametrize(
    "placed_on",
    [
        pytest.param(HOST, id="host"),
        pytest.param(accelerator(), marks=needs_an_accelerator, id="accelerator"),
    ],
)
@pytest.mark.parametrize("which", ["first", "middle", "last"])
@FUZZ
@given(slices=st.integers(1, 6), modes=st.integers(1, 8))
@example(slices=1, modes=1)
@example(slices=6, modes=8)
def test_a_slice_keeps_its_field_s_device(placed_on, which, slices, modes):
    coeff = torch.randn(2, slices, modes, dtype=DEFAULT_COEFF_DTYPE).to(placed_on)
    field = bench.FourierBasis(
        coeff=coeff, domain=((0.0, 1.0), (0.0, 1.0)), time_dependent=True
    )
    time_index = {"first": 0, "middle": slices // 2, "last": slices - 1}[which]

    slice_ = bench.spatial_field_at(field, time_index)

    assert slice_.coeff.device.type == placed_on.type
    assert torch.equal(slice_.coeff, coeff[:, time_index])


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

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def point_the_process_default_elsewhere(self, device):
        torch.set_default_device(device)

    @rule(seed=SEEDS, sizes=SIZES, device=st.sampled_from(DEVICES_PRESENT))
    def build_the_forced_case_at_other_sizes(self, seed, sizes, device):
        build_forced(seed, True, device, sizes)


BenchmarkCasesStayPut.TestCase.settings = settings(
    deadline=None, max_examples=10, stateful_step_count=10
)
TestBenchmarkCasesStayPut = pytest.mark.scenario(
    "A case does not depend on the process default"
)(
    pytest.mark.scenario("Adding a row leaves the others alone")(
        pytest.mark.oracle_invariant(BenchmarkCasesStayPut.TestCase)
    )
)


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


@pytest.mark.oracle_smoke
@FUZZ
@given(quick=st.booleans(), sizes=SIZES, device=st.sampled_from(DEVICES_PRESENT))
@example(quick=True, sizes=SMALLEST_SIZES, device=HOST)
@example(quick=False, sizes=FULL_SIZES, device=HOST)
def test_the_query_lies_on_the_output_time_grid_inside_the_domain(quick, sizes, device):
    case = build_forced(0, quick, device, sizes)
    times, space = case.query[:, 0].cpu(), case.query[:, 1].cpu()
    slices = case.split.train.u.coeff.shape[1]

    assert ((0 <= times) & (times <= 1)).all()
    assert times.max() == 1
    assert ((0 < times) & (times < 1)).any()
    assert ((0 <= space) & (space <= 1)).all()
    slice_index = times * (slices - 1)
    assert torch.equal(slice_index, slice_index.round())
    assert times.unique().numel() > 1
    assert space.unique().numel() > 1


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_quick_builds_a_smaller_case_than_full():
    quick = bench.burgers_forced_case(0, True, HOST)
    full = bench.burgers_forced_case(0, False, HOST)

    assert quick.split.train.n_functions < full.split.train.n_functions
    assert quick.split.test.n_functions < full.split.test.n_functions
    assert quick.split.train.sensors.shape[0] < full.split.train.sensors.shape[0]
    assert quick.split.train.u.coeff.shape[1] < full.split.train.u.coeff.shape[1]


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_the_forced_case_sizes_are_stated_by_name():
    with pytest.raises(TypeError, match="positional"):
        bench.burgers_forced_case(0, True, HOST, *SMALLEST_SIZES)


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


@pytest.mark.oracle_consistency
@pytest.mark.scenario("The end of the interval is scored")
@FUZZ
@given(seed=SEEDS, sizes=SIZES, device=st.sampled_from(DEVICES_PRESENT))
@example(seed=0, sizes=SMALLEST_SIZES, device=HOST)
@example(seed=2**32 - 1, sizes=QUICK_SIZES, device=HOST)
def test_the_end_of_the_interval_is_scored(seed, sizes, device):
    case = build_forced(seed, True, device, sizes)
    at_the_end = case.query[case.query[:, 0] == 1.0]

    assert len(at_the_end) > 0
    for half in (case.split.train, case.split.test):
        final_state = bench.FourierBasis(
            coeff=half.u.coeff[:, -1], domain=((0.0, 1.0),)
        )
        expected = final_state(at_the_end[:, 1:]).real.to(torch.float64)
        tolerance = END_OF_INTERVAL_RTOL * max(expected.abs().max().item(), 1e-12)
        assert (half.targets(at_the_end) - expected).abs().max() <= tolerance


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
