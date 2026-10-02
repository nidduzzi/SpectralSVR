import json
from pathlib import Path

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from SpectralSVR import FourierBasis, SpectralSVR
from SpectralSVR.model import LSSVR
from SpectralSVR.operators import OperatorDataset, SpectralSVROperator
from SpectralSVR.utils import to_complex_coeff

from tests._devices import DEVICES_PRESENT, HOST
from tests._time_varying import grid_points, rows, slice_times, time_varying_field

DOMAIN = ((0.0, 1.0), (0.0, 1.0))
ROUNDOFF_RTOL = 1e-6
INTERPOLATING_C = 1e6
PREDICTION_RTOL = 1e-5
FUZZ = settings(deadline=None, max_examples=20)
FIELDS = dict(
    seed=st.integers(0, 10_000),
    functions=st.integers(1, 4),
    slices=st.integers(1, 9),
    modes=st.integers(1, 9),
)


def interpolating_regressor():
    return LSSVR(kernel="linear", C=INTERPOLATING_C, dtype=torch.float32)


def seeded_draw(functions, sensors, seed):
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randn(sensors, functions, generator=generator, dtype=torch.float64)


def orthonormal_rows(columns):
    if torch.linalg.matrix_rank(columns) < columns.shape[1]:
        raise ValueError(
            f"a {tuple(columns.shape)} draw of rank below {columns.shape[1]} has no "
            "orthonormal rows spanning it"
        )
    q, r = torch.linalg.qr(columns)
    return (q * torch.sign(torch.diagonal(r))).T


def orthonormal_inputs(functions, sensors, seed):
    if functions > sensors:
        raise ValueError(
            f"{functions} orthonormal input functions cannot fit in {sensors} sensors"
        )
    return orthonormal_rows(seeded_draw(functions, sensors, seed))


def dataset_over(u, seed):
    sensors = torch.linspace(0, 1, 4, dtype=torch.float64).unsqueeze(-1)
    return OperatorDataset(
        sensors=sensors, f=orthonormal_inputs(len(u), len(sensors), seed), u=u
    )


def time_invariant_field(seed, functions, modes):
    generator = torch.Generator().manual_seed(seed)
    return FourierBasis(
        coeff=FourierBasis.generate_coeff(
            functions, (modes, modes), generator=generator
        ),
        domain=DOMAIN,
    )


@pytest.mark.oracle_analytic
@pytest.mark.scenario("Targets of a time-varying output are read at a time and a place")
@FUZZ
@given(**FIELDS, device=st.sampled_from(DEVICES_PRESENT))
@example(seed=0, functions=1, slices=1, modes=1, device=HOST)
@example(seed=0, functions=2, slices=5, modes=6, device=HOST)
@example(seed=0, functions=2, slices=9, modes=7, device=HOST)
def test_targets_of_a_time_varying_output_are_read_at_a_time_and_a_place(
    seed, functions, slices, modes, device
):
    slice_values, coeff = time_varying_field(seed, functions, slices, modes, device)
    u = FourierBasis(coeff=coeff, domain=DOMAIN, time_dependent=True)
    query = rows(slice_times(slices, device), grid_points(modes, device))

    targets = dataset_over(u, seed).targets(query)

    tolerance = ROUNDOFF_RTOL * slice_values.abs().max()
    assert (targets - slice_values.flatten(1)).abs().max() <= tolerance
    paired = FourierBasis.evaluate_time_dependent_at(coeff, query, domain=DOMAIN)
    assert torch.equal(targets, paired.real.to(torch.float64))


@pytest.mark.scenario("An oracle's accuracy does not depend on the luck of the draw")
@pytest.mark.scenario("A draw found to exceed the tolerance is pinned")
@pytest.mark.oracle_analytic
@pytest.mark.scenario(
    "Predictions of a time-varying output are read at the same points"
)
@FUZZ
@given(**FIELDS, query_device=st.sampled_from(DEVICES_PRESENT))
@example(seed=0, functions=1, slices=1, modes=1, query_device=HOST)
@example(seed=0, functions=4, slices=9, modes=9, query_device=HOST)
@example(seed=2211, functions=4, slices=1, modes=3, query_device=HOST)
@example(seed=253, functions=4, slices=1, modes=2, query_device=HOST)
def test_predictions_of_a_time_varying_output_are_read_at_the_same_points(
    seed, functions, slices, modes, query_device
):
    slice_values, coeff = time_varying_field(seed, functions, slices, modes, HOST)
    u = FourierBasis(coeff=coeff, domain=DOMAIN, time_dependent=True)
    dataset = dataset_over(u, seed)
    query = rows(slice_times(slices, query_device), grid_points(modes, query_device))

    fitted = SpectralSVROperator(FourierBasis(), interpolating_regressor()).fit(dataset)
    predicted = fitted.predict(dataset.f, query).cpu()

    assert predicted.shape == (functions, len(query))
    tolerance = PREDICTION_RTOL * slice_values.abs().max()
    assert (predicted - slice_values.flatten(1)).abs().max() <= tolerance


@pytest.mark.oracle_smoke
@pytest.mark.scenario(
    "A query point without a time is refused for a time-varying output"
)
@FUZZ
@given(**FIELDS)
@example(seed=0, functions=1, slices=2, modes=1)
def test_a_time_varying_output_refuses_a_point_without_a_time(
    seed, functions, slices, modes
):
    _, coeff = time_varying_field(seed, functions, slices, modes, HOST)
    u = FourierBasis(coeff=coeff, domain=DOMAIN, time_dependent=True)
    dataset = dataset_over(u, seed)
    space_only = grid_points(modes, HOST).unsqueeze(-1)
    fitted = SpectralSVROperator(FourierBasis(), interpolating_regressor()).fit(dataset)

    with pytest.raises(ValueError, match="time"):
        dataset.targets(space_only)
    with pytest.raises(ValueError, match="time"):
        fitted.predict(dataset.f, space_only)


@pytest.mark.oracle_smoke
@FUZZ
@given(
    seed=st.integers(0, 10_000),
    modes=st.integers(1, 6),
    stated_columns=st.sampled_from([1, 3]),
)
@example(seed=0, modes=1, stated_columns=1)
def test_a_time_invariant_output_refuses_rows_of_the_wrong_width(
    seed, modes, stated_columns
):
    u = time_invariant_field(seed, 2, modes)
    dataset = dataset_over(u, seed)
    model = SpectralSVR(FourierBasis(), interpolating_regressor())
    model.train(dataset.f, u.coeff)
    wrong_width = torch.full((4, stated_columns), 0.5, dtype=torch.float64)

    with pytest.raises(ValueError, match="shape"):
        model.forward(dataset.f, wrong_width, domain=DOMAIN)


@pytest.mark.oracle_consistency
@pytest.mark.scenario("A time-invariant output scores as before")
@FUZZ
@given(
    seed=st.integers(0, 10_000), functions=st.integers(1, 4), modes=st.integers(1, 9)
)
@example(seed=0, functions=1, modes=1)
def test_a_time_invariant_output_scores_as_before(seed, functions, modes):
    u = time_invariant_field(seed, functions, modes)
    dataset = dataset_over(u, seed)
    query = rows(grid_points(modes, HOST), grid_points(modes, HOST))
    model = SpectralSVR(FourierBasis(), interpolating_regressor())
    model.train(dataset.f, u.coeff)

    assert torch.equal(dataset.targets(query), u(query).real.to(torch.float64))
    predicted_coeff = model.regressor.predict(dataset.f)
    assert torch.equal(
        model.forward(dataset.f, query, domain=DOMAIN),
        FourierBasis.evaluate(
            coeff=to_complex_coeff(predicted_coeff).reshape(functions, modes, modes),
            x=query,
            domain=DOMAIN,
        ),
    )


@pytest.mark.oracle_smoke
@FUZZ
@given(**FIELDS)
@example(seed=0, functions=1, slices=1, modes=1)
def test_a_trained_model_reports_its_spatial_modes(seed, functions, slices, modes):
    _, coeff = time_varying_field(seed, functions, slices, modes, HOST)
    model = SpectralSVR(FourierBasis(), interpolating_regressor())
    model.train(
        dataset_over(FourierBasis(coeff=coeff, domain=DOMAIN), seed).f,
        coeff,
        u_time_dependent=True,
    )

    assert model.modes == (modes,)
    assert model.output_shape == (slices, modes)


class FittingAndPredicting(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.model = SpectralSVR(FourierBasis(), interpolating_regressor())
        self.dataset = None

    @initialize()
    def fit_the_field_that_once_exceeded_the_bound(self):
        self.fit(seed=2211, functions=4, slices=1, modes=3, varies_in_time=True)
        self.predictions_follow_the_most_recent_fit()

    @rule(
        seed=st.integers(0, 10_000),
        functions=st.integers(1, 4),
        slices=st.integers(1, 9),
        modes=st.integers(1, 7),
        varies_in_time=st.booleans(),
    )
    def fit(self, seed, functions, slices, modes, varies_in_time):
        if varies_in_time:
            _, coeff = time_varying_field(seed, functions, slices, modes, HOST)
            u = FourierBasis(coeff=coeff, domain=DOMAIN, time_dependent=True)
        else:
            u = time_invariant_field(seed, functions, modes)
        self.dataset = dataset_over(u, seed)
        self.model.train(self.dataset.f, u.coeff, u_time_dependent=u.time_dependent)
        self.query = rows(
            slice_times(slices if varies_in_time else modes, HOST),
            grid_points(modes, HOST),
        )

    @invariant()
    def predictions_follow_the_most_recent_fit(self):
        if self.dataset is None:
            return
        predicted = self.model.forward(self.dataset.f, self.query, domain=DOMAIN).real
        targets = self.dataset.targets(self.query)
        scale = max(targets.abs().max().item(), 1.0)
        assert (predicted - targets).abs().max() <= PREDICTION_RTOL * scale


FittingAndPredicting.TestCase.settings = settings(
    deadline=None, max_examples=15, stateful_step_count=8
)
TestFittingAndPredicting = pytest.mark.scenario(
    "An oracle's accuracy does not depend on the luck of the draw"
)(
    pytest.mark.scenario("A draw found to exceed the tolerance is pinned")(
        pytest.mark.oracle_invariant(FittingAndPredicting.TestCase)
    )
)


def classical_gram_schmidt(columns):
    basis = []
    for column in columns.T:
        residual = column - sum((column @ b) * b for b in basis)
        basis.append(residual / torch.linalg.vector_norm(residual))
    return torch.stack(basis)


@pytest.mark.oracle_independent
@pytest.mark.scenario("An oracle's accuracy does not depend on the luck of the draw")
@FUZZ
@given(seed=st.integers(0, 10_000), functions=st.integers(1, 4))
@example(seed=2211, functions=4)
@example(seed=253, functions=4)
@example(seed=0, functions=1)
def test_the_fixture_inputs_are_the_gram_schmidt_basis_of_their_seeded_draw(
    seed, functions
):
    inputs = orthonormal_inputs(functions, 4, seed)

    assert torch.allclose(
        inputs, classical_gram_schmidt(seeded_draw(functions, 4, seed)), atol=1e-10
    )
    assert torch.allclose(
        inputs @ inputs.T, torch.eye(functions, dtype=torch.float64), atol=1e-12
    )
    assert not torch.allclose(inputs, orthonormal_inputs(functions, 4, seed + 1))


@pytest.mark.oracle_smoke
@FUZZ
@given(functions=st.integers(1, 6), sensors=st.integers(1, 6), deficient=st.booleans())
@example(functions=5, sensors=4, deficient=False)
@example(functions=4, sensors=4, deficient=False)
@example(functions=2, sensors=4, deficient=True)
def test_the_fixture_refuses_inputs_it_cannot_make_orthonormal(
    functions, sensors, deficient
):
    if functions > sensors:
        with pytest.raises(
            ValueError,
            match=f"{functions} orthonormal input functions cannot fit in {sensors} sensors",
        ):
            orthonormal_inputs(functions, sensors, 0)
        return
    columns = seeded_draw(functions, sensors, 0)
    if deficient and functions > 1:
        columns[:, -1] = columns[:, 0]
        with pytest.raises(ValueError, match="no orthonormal rows"):
            orthonormal_rows(columns)
        return
    assert orthonormal_rows(columns).shape == (functions, sensors)


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_the_prediction_bound_sits_within_ten_times_the_recorded_worst():
    record = json.loads(
        (Path(__file__).parent / "prediction_error_record.json").read_text()
    )
    worst = max(record["worst_over_reruns"].values())

    assert worst < PREDICTION_RTOL <= 10 * worst
