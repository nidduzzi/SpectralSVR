import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from SpectralSVR import FourierBasis, accelerator

from tests._time_varying import grid_points, rows, slice_times, time_varying_field

DOMAIN = ((0.0, 1.0), (0.0, 1.0))
HOST = torch.device("cpu")
DEVICES_PRESENT = [HOST] + ([accelerator()] if torch.accelerator.is_available() else [])
FUZZ = settings(deadline=None, max_examples=30)
ROUNDOFF_RTOL = 1e-6
FIELDS = dict(
    seed=st.integers(0, 10_000),
    functions=st.integers(1, 3),
    slices=st.integers(1, 12),
    modes=st.integers(1, 12),
    device=st.sampled_from(DEVICES_PRESENT),
)


@pytest.mark.oracle_invariant
@FUZZ
@given(**FIELDS, times=st.lists(st.floats(0, 1), min_size=1, max_size=6))
@example(seed=0, functions=1, slices=1, modes=1, device=HOST, times=[0.0, 1.0])
@example(seed=0, functions=2, slices=5, modes=2, device=HOST, times=[0.0, 1.0, 0.125])
@example(seed=0, functions=2, slices=5, modes=7, device=HOST, times=[1.0, 0.5])
def test_paired_evaluation_equals_the_grid_form_at_every_pair(
    seed, functions, slices, modes, device, times
):
    slice_values, coeff = time_varying_field(seed, functions, slices, modes, device)
    time_axis = torch.tensor(times, dtype=torch.float64, device=device)
    space_axis = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64, device=device)

    grid = FourierBasis.evaluate(
        coeff, x=space_axis, t=time_axis, time_dependent=True, domain=DOMAIN
    )
    paired = FourierBasis.evaluate_time_dependent_at(
        coeff, rows(time_axis, space_axis), domain=DOMAIN
    )

    tolerance = ROUNDOFF_RTOL * slice_values.abs().max()
    assert (paired - grid.reshape(functions, -1)).abs().max() <= tolerance


@pytest.mark.oracle_analytic
@FUZZ
@given(**FIELDS)
@example(seed=0, functions=1, slices=1, modes=1, device=HOST)
@example(seed=0, functions=1, slices=2, modes=2, device=HOST)
@example(seed=0, functions=3, slices=12, modes=11, device=HOST)
def test_paired_evaluation_reads_slices_exactly_and_lerps_between_them(
    seed, functions, slices, modes, device
):
    slice_values, coeff = time_varying_field(seed, functions, slices, modes, device)
    times = slice_times(slices, device)
    space = grid_points(modes, device)
    tolerance = ROUNDOFF_RTOL * slice_values.abs().max()

    at_slices = FourierBasis.evaluate_time_dependent_at(
        coeff, rows(times, space), domain=DOMAIN
    ).real
    assert (at_slices - slice_values.flatten(1)).abs().max() <= tolerance

    if slices > 1:
        midway = (times[:-1] + times[1:]) / 2
        between = FourierBasis.evaluate_time_dependent_at(
            coeff, rows(midway, space), domain=DOMAIN
        ).real
        means = (slice_values[:, :-1] + slice_values[:, 1:]) / 2
        assert (between - means.flatten(1)).abs().max() <= tolerance


@pytest.mark.oracle_smoke
@FUZZ
@given(
    spatial_dimensions=st.integers(1, 3),
    points=st.integers(1, 5),
    stated_columns=st.sampled_from(["space only", "time too early", "time too late"]),
)
@example(spatial_dimensions=1, points=1, stated_columns="space only")
def test_a_point_without_a_time_in_the_interval_is_refused(
    spatial_dimensions, points, stated_columns
):
    modes = (3,) * spatial_dimensions
    coeff = torch.zeros(1, 4, *modes, dtype=torch.complex64)
    domain = ((0.0, 1.0),) * (1 + spatial_dimensions)
    place = torch.full((points, spatial_dimensions), 0.5, dtype=torch.float64)
    time = {"time too early": -0.25, "time too late": 1.25}.get(stated_columns)
    query = (
        place
        if time is None
        else torch.cat([torch.full((points, 1), time, dtype=torch.float64), place], 1)
    )

    with pytest.raises(ValueError, match="time"):
        FourierBasis.evaluate_time_dependent_at(coeff, query, domain=domain)
