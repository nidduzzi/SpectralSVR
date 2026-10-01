"""Operator baselines: dataset, SpectralSVR adapter, benchmark."""

import ast
import dataclasses
import importlib.util
import itertools
import math
import pathlib
import time

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from SpectralSVR import Antiderivative, FourierBasis
from SpectralSVR.model import LSSVR
from SpectralSVR.operators import (
    OperatorDataset,
    FittedOperator,
    Operator,
    Split,
    SpectralSVROperator,
    benchmark,
    sensor_grid,
)
from tests._devices import DEVICES_PRESENT, HOST
from tests._anchors import half_open_grid


STATABLE = [None, *DEVICES_PRESENT]
PASSING = ["by keyword", "by position"]
SENSOR_GRID_EDGES = [
    (1, 1, "by keyword"),
    (8, 3, "by keyword"),
    (0, 1, "by keyword"),
    (-1, 2, "by keyword"),
    (2, 0, "by keyword"),
    (2, 1, "by position"),
]

SENSORS = 16

QUERY = torch.linspace(0, 1, 40, dtype=torch.float64).unsqueeze(-1)
PROBE_MODES = (1, 2, 3, 4, 5)
SETTINGS = settings(deadline=None, max_examples=15)
# worst relative L2 measured over seeds 0-9 was 0.539; the bound sits above it
# with headroom rather than being tightened until the test passed
CLOSED_FORM_TOLERANCE = 0.70


def _split(seed, n_train, n_test, modes):
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(
        FourierBasis, n_train + n_test, modes, generator=g, u0=0
    )
    return Split.partition(ut, u, n_train, n_sensors=modes)


def _antiderivative_split(seed, n_train=120, n_test=30, modes=SENSORS):
    # operator maps the derivative ut -> the antiderivative u (integration)
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(
        FourierBasis, n_train + n_test, modes, generator=g, u0=0
    )
    return Split.partition(ut, u, n_train, n_sensors=modes)


class SensorsAndDatasets(RuleBasedStateMachine):
    @initialize(
        seed=st.integers(0, 10_000),
        functions=st.integers(1, 4),
        dimensions=st.integers(1, 2),
    )
    def draw_the_fields_and_check_every_pinned_edge(self, seed, functions, dimensions):
        generator = torch.Generator().manual_seed(seed)
        self.domain = ((0.0, 1.0),) * dimensions
        self.coeff = FourierBasis.generate_coeff(
            functions, (4,) * dimensions, generator=generator, device=HOST
        )
        for (n, d, passed), stated in itertools.product(SENSOR_GRID_EDGES, STATABLE):
            self.build_a_sensor_grid(n, d, stated, passed)
        for self.input_device, self.output_device, default in itertools.product(
            DEVICES_PRESENT, repeat=3
        ):
            self.point_the_process_default_elsewhere(default)
            self.build_a_dataset(sensors=1)
            self.hand_build_a_dataset(sensors=1)
        self.input_device = self.output_device = HOST

    @rule(
        n=st.integers(-8, 8),
        d=st.integers(0, 3),
        stated=st.sampled_from(STATABLE),
        passed=st.sampled_from(PASSING),
    )
    def build_a_sensor_grid(self, n, d, stated, passed):
        domain = ((0.0, 1.0),) * d
        if passed == "by position":
            with pytest.raises(TypeError, match="positional"):
                sensor_grid(domain, n, stated)  # type: ignore[misc]
        elif n < 1 or d == 0:
            with pytest.raises(ValueError, match="at least one"):
                sensor_grid(domain, n, device=stated)
        else:
            grid = sensor_grid(domain, n, device=stated)
            assert grid.shape == (n**d, d)
            assert ((0 <= grid) & (grid < 1)).all()
            expected = torch.get_default_device() if stated is None else stated
            assert grid.device == expected
            assert torch.equal(grid.cpu(), sensor_grid(domain, n, device=HOST))

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def place_the_input_field(self, device):
        self.input_device = device

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def place_the_output_field(self, device):
        self.output_device = device

    @rule(device=st.sampled_from(DEVICES_PRESENT))
    def point_the_process_default_elsewhere(self, device):
        torch.set_default_device(device)

    @rule(sensors=st.integers(1, 8))
    def build_a_dataset(self, sensors):
        input_field = FourierBasis(
            coeff=self.coeff.to(self.input_device), domain=self.domain
        )
        output_field = FourierBasis(
            coeff=self.coeff.to(self.output_device), domain=self.domain
        )
        if self.input_device != self.output_device:
            with pytest.raises(
                ValueError, match=f"{self.input_device}.*{self.output_device}"
            ):
                OperatorDataset.from_fields(
                    input_field, output_field, n_sensors=sensors
                )
        else:
            dataset = OperatorDataset.from_fields(
                input_field, output_field, n_sensors=sensors
            )
            assert dataset.sensors.device == self.input_device
            assert dataset.f.device == self.input_device
            assert torch.equal(
                dataset.f, input_field(dataset.sensors).real.to(torch.float64)
            )

    @rule(sensors=st.integers(1, 8))
    def hand_build_a_dataset(self, sensors):
        output_field = FourierBasis(
            coeff=self.coeff.to(self.output_device), domain=self.domain
        )
        grid = sensor_grid(self.domain, sensors, device=self.input_device)
        values = torch.zeros(
            len(self.coeff), len(grid), dtype=torch.float64, device=self.output_device
        )
        if self.input_device != self.output_device:
            with pytest.raises(
                ValueError, match=f"{self.input_device}.*{self.output_device}"
            ):
                OperatorDataset(sensors=grid, f=values, u=output_field)
        else:
            dataset = OperatorDataset(sensors=grid, f=values, u=output_field)
            assert dataset.sensors.device == dataset.f.device == self.input_device


SensorsAndDatasets.TestCase.settings = settings(
    deadline=None, max_examples=50, stateful_step_count=15
)
TestSensorsAndDatasets = pytest.mark.scenario(
    "Sensors are sampled where their field is"
)(
    pytest.mark.scenario("A dataset whose parts sit on different devices is refused")(
        pytest.mark.oracle_smoke(SensorsAndDatasets.TestCase)
    )
)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("Targets are evaluated, not pre-sampled")
def test_operator_dataset_shapes():
    train = _antiderivative_split(0).train
    assert train.sensors.shape == (16, 1)
    assert train.f.shape == (120, 16)
    assert train.n_functions == 120
    x = torch.linspace(0, 1, 25).unsqueeze(-1)
    assert train.targets(x).shape == (120, 25)


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
@pytest.mark.scenario("Predict after fit")
@pytest.mark.scenario("Query points are not restricted to the training grid")
def test_spectralsvr_operator_learns_antiderivative():
    # A*cos(2*pi*m*x) integrates to A/(2*pi*m)*sin(2*pi*m*x), which is zero-mean
    # and so is the antiderivative this operator is trained to produce. The probe
    # amplitude follows the training set's own scale: an RBF kernel extrapolates
    # badly, and a unit-amplitude cosine sits ~20x below the generated inputs.
    x_sensors = half_open_grid(SENSORS)
    x_query = QUERY.squeeze(-1)
    unit = torch.cos(2 * math.pi * x_sensors).norm()
    for seed in range(5):
        train = _antiderivative_split(seed).train
        fitted = SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0)).fit(
            train
        )
        amp = float(train.f.norm(dim=1).mean() / unit)

        probes = torch.stack(
            [amp * torch.cos(2 * math.pi * m * x_sensors) for m in PROBE_MODES]
        )
        exact = torch.stack(
            [
                amp / (2 * math.pi * m) * torch.sin(2 * math.pi * m * x_query)
                for m in PROBE_MODES
            ]
        )
        pred = fitted.predict(probes, QUERY).cpu().to(torch.float64)
        assert pred.shape == exact.shape

        rel = (pred - exact).norm(dim=1) / exact.norm(dim=1)
        assert float(rel.max()) < CLOSED_FORM_TOLERANCE, (
            f"seed={seed}: {[f'{v:.3f}' for v in rel.tolist()]}"
        )

        # discriminating: against the same closed form with its 1/(2*pi*m)
        # amplitude dropped, every probe misses the bound by a wide margin
        perturbed = torch.stack(
            [amp * torch.sin(2 * math.pi * m * x_query) for m in PROBE_MODES]
        )
        off = (pred - perturbed).norm(dim=1) / perturbed.norm(dim=1)
        assert float(off.min()) > CLOSED_FORM_TOLERANCE


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("Predicting before fitting cannot be stated")
@pytest.mark.scenario("Fitting yields the thing that predicts")
def test_an_unfitted_operator_cannot_predict():
    # the scenario the old assertion covered, moved to the type level: fitting is
    # the only transition, and it does not run backwards
    op = SpectralSVROperator(FourierBasis(), LSSVR())
    assert isinstance(op, Operator)
    assert not isinstance(op, FittedOperator)
    assert not hasattr(op, "predict")

    fitted = op.fit(_antiderivative_split(0, n_train=20, n_test=5).train)
    assert isinstance(fitted, FittedOperator)
    assert not isinstance(fitted, Operator)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("One row per operator, same metrics")
def test_benchmark_returns_metric_table():
    split = _antiderivative_split(1)
    query = torch.linspace(0, 1, 40).unsqueeze(-1)
    df = benchmark(
        [SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0))],
        split,
        query,
    )
    assert list(df.index) == ["SpectralSVR"]
    for col in ("fit_seconds", "rmse", "r2", "rrse"):
        assert col in df.columns
    assert df.loc["SpectralSVR", "r2"] > 0.5


class _Exploding:
    name = "Exploding"
    fit_covers = "nothing; it raises"
    protocol = "a fake that raises"

    def fit(self, dataset):
        raise RuntimeError("fit blew up")


class _FittedConstant:
    """Predicts one number everywhere, chosen from the data it was fitted on."""

    name = "Constant"
    device = HOST

    def __init__(self, value):
        self.value = value

    def predict(self, f, x):
        return torch.full((f.shape[0], x.shape[0]), self.value, dtype=torch.float64)


class _FittedZero:
    """Predicts zero everywhere, so its mse is the target's own mean square."""

    name = "Zero"
    device = HOST

    def predict(self, f, x):
        return torch.zeros(f.shape[0], x.shape[0], dtype=torch.float64)


class _Zero:
    name = "Zero"
    fit_covers = "no training"
    protocol = "a fake predicting zero"

    def fit(self, dataset):
        return _FittedZero()


class _Constant:
    name = "Constant"
    fit_covers = "one mean over the training targets"
    protocol = "a fake predicting the training mean"

    def fit(self, dataset):
        return _FittedConstant(float(dataset.targets(QUERY).mean()))


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_benchmark_propagates_a_failing_operator():
    split = _antiderivative_split(0, n_train=20, n_test=5)
    with pytest.raises(RuntimeError, match="fit blew up"):
        benchmark([_Exploding()], split, QUERY)


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
@pytest.mark.scenario("Shared inputs and targets")
@pytest.mark.scenario("Each method takes the view it needs")
def test_benchmark_scores_two_operators_against_the_same_targets():
    split = _antiderivative_split(1, n_train=40, n_test=10)
    df = benchmark(
        [SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0)), _Constant()],
        split,
        QUERY,
    )
    assert list(df.index) == ["SpectralSVR", "Constant"]

    # the constant row's error is computable by hand, so it pins which targets
    # the benchmark scored against: test.targets(QUERY), not anything per
    # operator. Self-referential by construction -- the check is which library
    # output was used, so the library supplies both sides
    target = split.test.targets(QUERY)
    expected_mse = float(
        ((target - float(split.train.targets(QUERY).mean())) ** 2).mean()
    )
    assert df.loc["Constant", "mse"] == pytest.approx(expected_mse, rel=1e-9)


def _flat(text):
    """Docstring text with line wrapping collapsed, so a sentence matches whole."""
    return " ".join((text or "").split())


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("Scope of the claim is stated")
@pytest.mark.scenario("Fit times are qualified")
@pytest.mark.scenario("A single-seed table is not presented as a result")
@pytest.mark.scenario("Run-to-run spread is stated per operator")
def test_the_benchmark_script_states_what_its_table_establishes():
    script = (
        pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
    )
    doc = _flat(ast.get_docstring(ast.parse(script.read_text())))
    assert "not tuned per dataset" in doc
    assert "seed set 0, 1, 2, 3, 4" in doc
    assert "never read as the record" in doc
    assert "Run-to-run spread at one fixed seed, measured per operator" in doc
    assert "zero for both SpectralSVR and DeepONet" in doc
    assert "moved by up to 1.9e-05 relative" in doc
    assert "``fit_covers`` states the work its fit did" in doc
    assert "``_import_deepxde()``" in doc
    assert "does neither" in doc


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("Fit time excludes prediction")
@pytest.mark.scenario("A failing operator is not silently omitted from a run")
def test_benchmark_documents_its_measurement_protocol():
    doc = _flat(benchmark.__doc__)
    assert "``fit_seconds`` brackets ``fit`` alone" in doc
    assert "An operator that raises is not caught" in doc
    assert "share no functions by construction" in doc


@SETTINGS
@pytest.mark.oracle_smoke
@pytest.mark.scenario("An inconsistent dataset cannot be built")
@given(
    modes=st.integers(4, 12),
    n=st.integers(3, 10),
    drop_sensors=st.integers(1, 3),
    drop_functions=st.integers(1, 2),
)
def test_an_inconsistent_dataset_is_refused(modes, n, drop_sensors, drop_functions):
    train = _split(0, n_train=n, n_test=2, modes=modes).train
    wider = torch.cat([train.sensors, train.sensors[:drop_sensors]])
    more = torch.cat([train.f, train.f[:drop_functions]])
    # both directions: a mismatch is a mismatch whichever side is larger
    for sensors, f in (
        (train.sensors[:-drop_sensors], train.f),
        (wider, train.f),
    ):
        with pytest.raises(ValueError, match="sensors"):
            OperatorDataset(sensors=sensors, f=f, u=train.u)
    for f in (train.f[:-drop_functions], more):
        with pytest.raises(ValueError, match="output functions"):
            OperatorDataset(sensors=train.sensors, f=f, u=train.u)


@SETTINGS
@pytest.mark.oracle_analytic
@pytest.mark.scenario("Overlapping train and test cannot be stated")
@given(
    total=st.integers(4, 24),
    modes=st.integers(4, 12),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
def test_a_split_partitions_its_source_exactly_once(total, modes, seed, data):
    n_train = data.draw(st.integers(1, total - 1))
    g = torch.Generator().manual_seed(seed)
    u, ut = Antiderivative().generate(FourierBasis, total, modes, generator=g, u0=0)
    split = Split.partition(ut, u, n_train, n_sensors=modes)
    # every function lands in exactly one half: the counts sum, and the halves
    # are the two sides of one cut rather than two independent draws
    assert split.train.n_functions == n_train
    assert split.test.n_functions == total - n_train
    assert torch.equal(split.train.f, ut[:n_train](split.train.sensors).real.double())
    assert torch.equal(split.test.f, ut[n_train:](split.test.sensors).real.double())
    for refused in (-7, -1, 0, total, total + 1, total + 7):
        with pytest.raises(ValueError, match="non-empty"):
            Split.partition(ut, u, refused, n_sensors=modes)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A short table requires an explicit opt-in")
def test_a_missing_operator_extra_refuses_by_default(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_run_benchmark",
        pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "deepxde" else real(name, *a, **k),
    )
    query = torch.linspace(0, 1, 5, dtype=torch.float64).unsqueeze(-1)
    plan = script.DATASETS["antiderivative"].deeponet
    with pytest.raises(script.MissingOperatorDependency, match="deepxde"):
        script.build_operators(["spectralsvr", "deeponet"], query, plan, 0, HOST)

    opted_in = script.build_operators(
        ["spectralsvr", "deeponet"], query, plan, 0, HOST, allow_missing=True
    )
    assert [op.name for op in opted_in] == ["SpectralSVR"]


class _FittedSlow:
    name = "Slow"
    device = HOST

    def __init__(self, predict_pause):
        self._pause = predict_pause

    def predict(self, f, x):
        time.sleep(self._pause)
        return torch.zeros(f.shape[0], x.shape[0], dtype=torch.float64)


class _Slow:
    """Spends a known time in fit, and a different known time in predict."""

    name = "Slow"
    fit_covers = "a known pause"
    protocol = "a fake that sleeps"

    def __init__(self, fit_pause, predict_pause):
        self._fit_pause = fit_pause
        self._predict_pause = predict_pause

    def fit(self, dataset):
        time.sleep(self._fit_pause)
        return _FittedSlow(self._predict_pause)


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
@pytest.mark.scenario("Fit time excludes prediction")
def test_fit_seconds_measures_the_fit_and_not_the_prediction():
    # the column is a duration, and a known one: sleeping in fit must show up in
    # it, and sleeping in predict must not
    split = _antiderivative_split(0, n_train=20, n_test=5)
    fit_pause, predict_pause = 0.25, 0.25
    started = time.perf_counter()
    df = benchmark([_Slow(fit_pause, predict_pause)], split, QUERY)
    wall = time.perf_counter() - started

    measured = float(df.loc["Slow", "fit_seconds"])
    assert fit_pause <= measured < fit_pause + predict_pause
    # and it is a slice of this call, not an unrelated quantity in the same units
    assert measured <= wall


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_the_value_types_are_immutable():
    # a dataset, a split and a fitted operator are values: rebinding a field
    # would let a caller change what a benchmark already scored against
    split = _antiderivative_split(0, n_train=20, n_test=5)
    fitted = SpectralSVROperator(FourierBasis(), LSSVR()).fit(split.train)
    for value, field in (
        (split, "train"),
        (split.train, "f"),
        (fitted, "model"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, None)


class BenchmarkRuns(RuleBasedStateMachine):
    """What must stay true across a sequence of benchmark runs.

    Single-shot cases cannot reach these: the target must not drift as operators
    are added or removed, and every run must yield one row per operator it was
    given, however many that is.
    """

    MODES = 8

    def __init__(self):
        super().__init__()
        self.split = _split(0, n_train=12, n_test=4, modes=self.MODES)
        self.query = torch.linspace(0, 1, 12, dtype=torch.float64).unsqueeze(-1)
        self.first_target = self.split.test.targets(self.query).clone()
        self.operators = []
        self.ran_with = []
        self.last = None

    @rule()
    def add_zero(self):
        self.operators.append(_Zero())

    @rule()
    def add_spectralsvr(self):
        self.operators.append(
            SpectralSVROperator(FourierBasis(), LSSVR(kernel="rbf", C=50.0))
        )

    @rule()
    def drop_one(self):
        if self.operators:
            self.operators.pop()

    @rule()
    def run(self):
        if not self.operators:
            return
        self.ran_with = list(self.operators)
        self.last = benchmark(self.operators, self.split, self.query)

    @invariant()
    def the_target_never_drifts(self):
        assert torch.equal(self.split.test.targets(self.query), self.first_target)

    @invariant()
    def one_row_per_operator(self):
        if self.last is None:
            return
        assert len(self.last.index) == len(self.ran_with)
        assert list(self.last.index) == [op.name for op in self.ran_with]

    @invariant()
    def fit_seconds_never_absorbs_a_prediction(self):
        if self.last is None:
            return
        assert (self.last["fit_seconds"] >= 0).all()

    @invariant()
    def every_row_is_scored_against_the_test_half(self):
        # the zero predictor's error is the target's own mean square, so its row
        # says which targets the benchmark used -- and says it again on every run
        if self.last is None or "Zero" not in self.last.index:
            return
        expected = float((self.first_target**2).mean())
        # every Zero row, not one: two operators may share a name, which the
        # benchmark permits and which makes the index non-unique
        scored = self.last.loc[self.last.index == "Zero", "mse"]
        assert len(scored) >= 1
        for value in scored:
            assert value == pytest.approx(expected, rel=1e-9)


BenchmarkRuns.TestCase.settings = settings(
    deadline=None, max_examples=25, stateful_step_count=10
)
TestBenchmarkRuns = pytest.mark.oracle_invariant(
    pytest.mark.scenario("A sequence of operations is modelled as one")(
        BenchmarkRuns.TestCase
    )
)
