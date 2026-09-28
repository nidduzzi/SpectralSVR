import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from SpectralSVR import FourierBasis, accelerator
from SpectralSVR.utils import to_real_coeff

from tests._time_varying import (
    UNIT,
    grid_points,
    rows,
    slice_times,
    time_varying_field,
)

HOST = torch.device("cpu")
DEVICES_PRESENT = [HOST] + ([accelerator()] if torch.accelerator.is_available() else [])
FUZZ = settings(deadline=None, max_examples=30)
ROUNDOFF_RTOL = 1e-6
INTERVALS = st.tuples(st.floats(-2, 2), st.floats(0.25, 3)).map(
    lambda start_and_length: (start_and_length[0], sum(start_and_length))
)
SHIFTED_TIME = (-0.5, 1.75)
SHIFTED_SPACE = (0.25, 2.25)
FIELDS = dict(
    seed=st.integers(0, 10_000),
    functions=st.integers(1, 3),
    slices=st.integers(1, 12),
    modes=st.integers(1, 12),
    device=st.sampled_from(DEVICES_PRESENT),
    time_interval=INTERVALS,
    space_interval=INTERVALS,
)


@pytest.mark.oracle_invariant
@FUZZ
@given(**FIELDS, fractions=st.lists(st.floats(0, 1), min_size=1, max_size=6))
@example(
    seed=0,
    functions=1,
    slices=1,
    modes=1,
    device=HOST,
    time_interval=UNIT,
    space_interval=UNIT,
    fractions=[0.0, 1.0],
)
@example(
    seed=0,
    functions=2,
    slices=5,
    modes=2,
    device=HOST,
    time_interval=SHIFTED_TIME,
    space_interval=SHIFTED_SPACE,
    fractions=[0.0, 1.0, 0.125],
)
@example(
    seed=0,
    functions=2,
    slices=5,
    modes=7,
    device=HOST,
    time_interval=SHIFTED_TIME,
    space_interval=UNIT,
    fractions=[1.0, 0.5],
)
def test_paired_evaluation_equals_the_grid_form_at_every_pair(
    seed, functions, slices, modes, device, time_interval, space_interval, fractions
):
    slice_values, coeff = time_varying_field(seed, functions, slices, modes, device)
    domain = (time_interval, space_interval)
    t_start, t_stop = time_interval
    x_start, x_stop = space_interval
    time_axis = t_start + (t_stop - t_start) * torch.tensor(
        fractions, dtype=torch.float64, device=device
    )
    space_axis = x_start + (x_stop - x_start) * torch.tensor(
        [0.0, 0.3, 1.0], dtype=torch.float64, device=device
    )
    query = rows(time_axis, space_axis)

    grid = FourierBasis.evaluate(
        coeff, x=space_axis, t=time_axis, time_dependent=True, domain=domain
    )
    paired = FourierBasis.evaluate_time_dependent_at(coeff, query, domain=domain)

    tolerance = ROUNDOFF_RTOL * slice_values.abs().max()
    assert (paired - grid.reshape(functions, -1)).abs().max() <= tolerance
    from_real_coeff = FourierBasis.evaluate_time_dependent_at(
        to_real_coeff(coeff), query, domain=domain
    )
    assert torch.equal(from_real_coeff, paired)


@pytest.mark.oracle_analytic
@FUZZ
@given(**FIELDS)
@example(
    seed=0,
    functions=1,
    slices=1,
    modes=1,
    device=HOST,
    time_interval=UNIT,
    space_interval=UNIT,
)
@example(
    seed=0,
    functions=1,
    slices=2,
    modes=2,
    device=HOST,
    time_interval=SHIFTED_TIME,
    space_interval=SHIFTED_SPACE,
)
@example(
    seed=0,
    functions=3,
    slices=12,
    modes=11,
    device=HOST,
    time_interval=SHIFTED_TIME,
    space_interval=UNIT,
)
def test_paired_evaluation_reads_slices_exactly_and_lerps_between_them(
    seed, functions, slices, modes, device, time_interval, space_interval
):
    slice_values, coeff = time_varying_field(seed, functions, slices, modes, device)
    domain = (time_interval, space_interval)
    times = slice_times(slices, device, time_interval)
    space = grid_points(modes, device, space_interval)
    tolerance = ROUNDOFF_RTOL * slice_values.abs().max()

    at_slices = FourierBasis.evaluate_time_dependent_at(
        coeff, rows(times, space), domain=domain
    ).real
    assert (at_slices - slice_values.flatten(1)).abs().max() <= tolerance

    if slices > 1:
        midway = (times[:-1] + times[1:]) / 2
        between = FourierBasis.evaluate_time_dependent_at(
            coeff, rows(midway, space), domain=domain
        ).real
        means = (slice_values[:, :-1] + slice_values[:, 1:]) / 2
        assert (between - means.flatten(1)).abs().max() <= tolerance


@pytest.mark.oracle_invariant
@FUZZ
@given(
    seed=st.integers(0, 10_000),
    slices=st.integers(1, 6),
    modes=st.tuples(st.integers(1, 5), st.integers(1, 5)),
    fractions=st.lists(st.floats(0, 1), min_size=1, max_size=4),
)
@example(seed=0, slices=1, modes=(1, 1), fractions=[0.0, 1.0])
@example(seed=0, slices=4, modes=(2, 3), fractions=[1.0, 0.5])
def test_paired_evaluation_reads_two_spatial_axes_like_the_grid_form(
    seed, slices, modes, fractions
):
    generator = torch.Generator().manual_seed(seed)
    slice_values = torch.randn(2 * slices, *modes, generator=generator)
    coeff = FourierBasis.transform(slice_values).reshape(2, slices, *modes)
    domain = (SHIFTED_TIME, UNIT, SHIFTED_SPACE)
    t_start, t_stop = SHIFTED_TIME
    times = t_start + (t_stop - t_start) * torch.tensor(fractions, dtype=torch.float64)
    places = torch.tensor([[0.0, 0.25], [0.5, 1.5], [1.0, 2.25]], dtype=torch.float64)
    query = torch.cat(
        [
            times.repeat_interleave(len(places)).unsqueeze(-1),
            places.repeat(len(times), 1),
        ],
        dim=1,
    )

    grid = FourierBasis.evaluate(
        coeff, x=places, t=times, time_dependent=True, domain=domain
    )
    paired = FourierBasis.evaluate_time_dependent_at(coeff, query, domain=domain)

    tolerance = ROUNDOFF_RTOL * slice_values.abs().max()
    assert (paired - grid.reshape(2, -1)).abs().max() <= tolerance


@pytest.mark.oracle_consistency
@FUZZ
@given(
    seed=st.integers(0, 10_000),
    slices=st.integers(1, 6),
    modes=st.integers(1, 8),
    fractions=st.lists(st.floats(0, 1), min_size=1, max_size=4),
)
@example(seed=0, slices=1, modes=1, fractions=[0.0, 1.0])
def test_an_unstated_domain_is_the_unit_interval_on_every_axis(
    seed, slices, modes, fractions
):
    _, coeff = time_varying_field(seed, 2, slices, modes, HOST)
    query = rows(torch.tensor(fractions, dtype=torch.float64), grid_points(modes, HOST))

    unstated = FourierBasis.evaluate_time_dependent_at(coeff, query)
    stated = FourierBasis.evaluate_time_dependent_at(coeff, query, domain=(UNIT, UNIT))

    assert torch.equal(unstated, stated)


@pytest.mark.oracle_smoke
@FUZZ
@given(
    spatial_dimensions=st.integers(1, 3),
    points=st.integers(1, 5),
    stated_columns=st.sampled_from(
        [
            "space only",
            "an extra column",
            "time too early",
            "time too late",
            "flat",
            "stacked",
        ]
    ),
)
@example(spatial_dimensions=1, points=1, stated_columns="space only")
@example(spatial_dimensions=2, points=2, stated_columns="an extra column")
@example(spatial_dimensions=1, points=2, stated_columns="stacked")
def test_a_point_without_a_time_in_the_interval_is_refused(
    spatial_dimensions, points, stated_columns
):
    modes = (3,) * spatial_dimensions
    coeff = torch.zeros(1, 4, *modes, dtype=torch.complex64)
    domain = (UNIT,) * (1 + spatial_dimensions)
    place = torch.full((points, spatial_dimensions), 0.5, dtype=torch.float64)

    def at_time(time):
        return torch.cat([torch.full((points, 1), time, dtype=torch.float64), place], 1)

    query = {
        "space only": place,
        "an extra column": torch.cat([at_time(0.5), place[:, :1]], 1),
        "time too early": at_time(-0.25),
        "time too late": at_time(1.25),
        "flat": at_time(0.5).flatten(),
        "stacked": at_time(0.5).unsqueeze(0),
    }[stated_columns]

    with pytest.raises(ValueError, match="time"):
        FourierBasis.evaluate_time_dependent_at(coeff, query, domain=domain)
