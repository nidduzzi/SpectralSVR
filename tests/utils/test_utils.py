import numpy as np
import pytest
import warnings
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from SpectralSVR.utils import (
    DEFAULT_METRICS,
    accelerator,
    ExactModeCount,
    StandardScaler,
    WavenumberCutoff,
    euler_solver,
    get_metrics,
    interpolate_tensor,
    refuse_a_generator_placed_elsewhere,
    resize_modes,
    resolve_device,
    scale_to_standard,
    from_mag_angle,
    to_complex_coeff,
    to_mag_angle,
    to_real_coeff,
    zero_coeff,
)

SETTINGS = settings(
    deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow]
)
_elements = st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False)


@st.composite
def real_arrays(draw):
    # to_complex_coeff pairs the last axis, so its width must be even.
    rows = draw(st.integers(1, 12))
    half = draw(st.integers(1, 16))
    return torch.from_numpy(
        draw(hnp.arrays(np.float64, (rows, 2 * half), elements=_elements))
    )


@pytest.mark.oracle_consistency
@SETTINGS
@given(a=real_arrays())
def test_real_complex_real_roundtrip(a):
    # real -> complex -> real is a clean round trip for even width
    assert torch.allclose(to_real_coeff(to_complex_coeff(a)), a)


@pytest.mark.oracle_independent
@SETTINGS
@given(
    rows=st.integers(1, 8),
    half=st.integers(1, 8),
    seed=st.integers(0, 10_000),
)
def test_coeff_interleave_semantics(rows, half, seed):
    # Independent anchor (not a round trip): pin the actual re/im layout against
    # separately-built real and imaginary parts. A swapped interleave would pass
    # the two round-trip tests but fails here.
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(rows, half, generator=g)
    im = torch.randn(rows, half, generator=g)
    interleaved = torch.empty(rows, 2 * half)
    interleaved[:, 0::2] = re  # even slots = real, odd = imag
    interleaved[:, 1::2] = im
    c = to_complex_coeff(interleaved)
    assert torch.allclose(c.real, re) and torch.allclose(c.imag, im)
    assert torch.allclose(to_real_coeff(c), interleaved)


@pytest.mark.oracle_consistency
@SETTINGS
@given(
    rows=st.integers(1, 12),
    cols=st.integers(1, 16),
    seed=st.integers(0, 10_000),
)
def test_complex_invertible(rows, cols, seed):
    c = torch.randn(
        rows, cols, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    assert torch.equal(to_complex_coeff(to_real_coeff(c)), c)


STATABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.oracle_smoke
@pytest.mark.scenario("A stated device is used")
@SETTINGS
@given(name=st.sampled_from(STATABLE_DEVICES))
def test_resolve_device_uses_the_stated_device(name):
    assert resolve_device(name) == torch.device(name)
    assert resolve_device(torch.device(name)) == torch.device(name)


@pytest.mark.oracle_independent
@pytest.mark.scenario("A process-wide setting is followed")
@SETTINGS
@given(name=st.sampled_from(STATABLE_DEVICES))
def test_resolve_device_follows_the_process_default(name):
    torch.set_default_device(name)
    assert resolve_device(None) == torch.zeros(1).device


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("Naming the accelerator is a statement, not a search")
def test_accelerator_names_whichever_one_is_present():
    named = accelerator()
    if torch.accelerator.is_available():
        assert named == torch.accelerator.current_accelerator()
        assert torch.zeros(1, device=named).device.type == named.type
    else:
        assert named == torch.get_default_device()
    assert resolve_device(named) == named


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("The host is not overridden")
def test_a_stated_host_survives_an_available_accelerator():
    if not torch.accelerator.is_available():
        pytest.skip(
            "an accelerator must be present for the host to be overridable at all"
        )
    torch.set_default_device("cpu")
    assert resolve_device(None) == torch.device("cpu")


DISAGREEING_PAIRS = (
    [(a, b) for a in STATABLE_DEVICES for b in STATABLE_DEVICES if a != b]
    or None
)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A stated index is part of the statement")
def test_two_stated_indices_that_differ_are_a_disagreement():
    if not torch.cuda.is_available():
        pytest.skip("an indexed accelerator is needed to state two of them")
    with pytest.raises(RuntimeError, match="move the source"):
        refuse_a_generator_placed_elsewhere(torch.Generator(device="cuda:0"), "cuda:1")


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A stated index is part of the statement")
def test_an_unstated_index_agrees_with_a_stated_one():
    if not torch.cuda.is_available():
        pytest.skip("an indexed accelerator is needed for an index to be unstated")
    refuse_a_generator_placed_elsewhere(torch.Generator(device="cuda:0"), "cuda")
    refuse_a_generator_placed_elsewhere(torch.Generator(device="cuda"), "cuda:0")


@pytest.mark.oracle_smoke
@pytest.mark.scenario("A mismatched random source names both devices")
@SETTINGS
@given(pair=st.sampled_from(DISAGREEING_PAIRS or [("cpu", "cpu")]))
def test_a_misplaced_random_source_names_both_devices(pair):
    if DISAGREEING_PAIRS is None:
        pytest.skip("two device types must be present for them to disagree")
    source_name, work_name = pair
    source = torch.Generator(device=source_name)
    with pytest.raises(RuntimeError) as refusal:
        refuse_a_generator_placed_elsewhere(source, work_name)
    message = str(refusal.value)
    assert str(source.device) in message
    assert str(resolve_device(work_name)) in message
    assert "move the source" in message


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.scenario("A mismatched random source names both devices")
def test_a_random_source_device_is_invisible_to_a_type_checker():
    host = torch.Generator()
    assert type(host) is type(torch.Generator(device=accelerator()))
    if torch.accelerator.is_available():
        assert host.device != accelerator()


@pytest.mark.oracle_smoke
@pytest.mark.scenario("An agreeing random source is unremarkable")
@SETTINGS
@given(name=st.sampled_from(STATABLE_DEVICES))
def test_an_agreeing_random_source_is_unremarkable(name):
    torch.set_default_device(name)
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        refuse_a_generator_placed_elsewhere(torch.Generator(device=name), name)
        refuse_a_generator_placed_elsewhere(torch.Generator(device=name), None)
        refuse_a_generator_placed_elsewhere(None, name)
    assert [str(w.message) for w in raised] == []


@pytest.mark.oracle_smoke
@SETTINGS
@given(rows=st.integers(2, 20), cols=st.integers(1, 5), seed=st.integers(0, 10_000))
def test_get_metrics_default_keys(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    preds = torch.randn(rows, cols, generator=g)
    targets = torch.randn(rows, cols, generator=g)
    m = get_metrics(preds, targets)
    assert set(m) == set(DEFAULT_METRICS) | {"pred_nan_sum"}
    assert m["mse"] >= 0.0


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_get_metrics_single_sample_r2_is_nan():
    m = get_metrics(torch.randn(1, 3), torch.randn(1, 3))
    assert m["r2"] != m["r2"]  # NaN


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_get_metrics_injected_metric_set():
    m = get_metrics(
        torch.zeros(4, 2),
        torch.zeros(4, 2),
        metrics={"custom": lambda p, t: (p - t).abs().sum()},
    )
    assert m["custom"] == 0.0
    assert "mse" not in m
    assert "pred_nan_sum" in m


# --------------------------------------------------------------------------- #
# StandardScaler
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_consistency
@given(rows=st.integers(2, 40), cols=st.integers(1, 8), seed=st.integers(0, 10_000))
@SETTINGS
def test_standard_scaler_roundtrip_real(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g) * 3 + 1.5
    scaler = StandardScaler().fit(x)
    assert torch.allclose(scaler.inverse(scaler.transform(x)), x, atol=1e-4)


@pytest.mark.oracle_invariant
@given(rows=st.integers(2, 40), cols=st.integers(1, 8), seed=st.integers(0, 10_000))
@SETTINGS
def test_standard_scaler_standardizes(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g) * 3 + 1.5
    xt = StandardScaler().fit(x).transform(x)
    assert torch.allclose(xt.mean(0), torch.zeros(cols), atol=1e-4)
    assert torch.allclose(xt.std(0, unbiased=False), torch.ones(cols), atol=1e-3)


@pytest.mark.oracle_consistency
@given(rows=st.integers(2, 30), cols=st.integers(1, 6), seed=st.integers(0, 10_000))
@SETTINGS
def test_standard_scaler_roundtrip_complex(rows, cols, seed):
    c = torch.randn(
        rows, cols, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    scaler = StandardScaler().fit(c)
    assert torch.allclose(scaler.inverse(scaler.transform(c)), c, atol=1e-3)


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_standard_scaler_tuple_and_subset():
    a = torch.randn(10, 3)
    b = torch.randn(10, 2, dtype=torch.complex64)
    scaler = StandardScaler().fit((a, b))
    ta, tb = scaler.transform((a, b))
    assert ta.shape == a.shape and tb.shape == b.shape
    sub = scaler.get_subset_scaler(0)
    assert torch.allclose(sub.inverse(sub.transform(a)), a, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_standard_scaler_consistency_checks():
    scaler = StandardScaler().fit(torch.randn(8, 3))
    with pytest.raises(AssertionError):
        scaler.transform(torch.randn(8, 4))  # wrong 2nd dim
    with pytest.raises(AssertionError):
        scaler.transform(torch.randn(8, 3, dtype=torch.complex64))  # complex mismatch


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_standard_scaler_save_load(tmp_path):
    x = torch.randn(12, 4)
    scaler = StandardScaler().fit(x)
    path = str(tmp_path / "scaler.pt")
    scaler.save(path)
    loaded = StandardScaler.load(path)
    assert torch.allclose(loaded.transform(x), scaler.transform(x), atol=1e-6)


@pytest.mark.oracle_invariant
@given(rows=st.integers(2, 30), cols=st.integers(1, 6), seed=st.integers(0, 10_000))
@SETTINGS
def test_scale_to_standard(rows, cols, seed):
    x = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))
    xt = scale_to_standard(x)
    assert torch.allclose(xt.mean(0), torch.zeros(cols), atol=1e-4)


# --------------------------------------------------------------------------- #
# to_mag_angle
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_analytic
@given(rows=st.integers(1, 20), cols=st.integers(1, 10), seed=st.integers(0, 10_000))
@SETTINGS
def test_to_mag_angle_reconstructs(rows, cols, seed):
    c = torch.randn(
        rows, cols, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    ma = to_mag_angle(c)
    assert ma.shape == (rows, 2 * cols)
    mag, angle = ma[:, ::2], ma[:, 1::2]
    assert torch.allclose(mag * torch.exp(1j * angle), c, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_to_mag_angle_idempotent_on_real():
    r = torch.randn(3, 4)
    assert torch.equal(to_mag_angle(r), r)


POLAR_SHAPES = ((5, 9), (2, 8, 8), (2, 4, 4, 4))


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_to_mag_angle_handles_any_number_of_mode_axes():
    generator = torch.Generator().manual_seed(11)
    for shape in POLAR_SHAPES:
        c = torch.randn(*shape, dtype=torch.complex64, generator=generator)
        ma = to_mag_angle(c)
        assert ma.shape == (*shape[:-1], shape[-1] * 2)
        mag, angle = ma[..., ::2], ma[..., 1::2]
        assert torch.allclose(mag * torch.exp(1j * angle), c, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_from_mag_angle_inverts_to_mag_angle():
    generator = torch.Generator().manual_seed(12)
    for shape in POLAR_SHAPES:
        c = torch.randn(*shape, dtype=torch.complex64, generator=generator)
        assert torch.allclose(from_mag_angle(to_mag_angle(c)), c, atol=1e-4)


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_polar_encoding_precision_corresponds():
    for complex_dtype, real_dtype in (
        (torch.complex64, torch.float32),
        (torch.complex128, torch.float64),
    ):
        c = torch.randn(3, 4, dtype=complex_dtype)
        assert to_mag_angle(c).dtype == real_dtype
        assert from_mag_angle(to_mag_angle(c)).dtype == complex_dtype


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no accelerator available")
def test_polar_encoding_preserves_device():
    c = torch.randn(3, 4, dtype=torch.complex64).cuda()
    assert to_mag_angle(c).device.type == "cuda"
    assert from_mag_angle(to_mag_angle(c)).device.type == "cuda"


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_from_mag_angle_idempotent_on_complex():
    c = torch.randn(3, 4, dtype=torch.complex64)
    assert torch.equal(from_mag_angle(c), c)


# --------------------------------------------------------------------------- #
# euler_solver (MMS: dy/dt = -a y -> y0 exp(-a t))
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_analytic
@given(a=st.floats(0.2, 3.0), seed=st.integers(0, 10_000))
@SETTINGS
def test_euler_solver_linear_decay(a, seed):
    y0 = torch.randn(2, generator=torch.Generator().manual_seed(seed))
    t = torch.linspace(0, 1.0, 2000)

    def rhs(ti, y):
        return -a * y

    sol = euler_solver(rhs, y0, t)
    exact = y0 * torch.exp(-a * t[-1])
    assert torch.allclose(sol[-1], exact, atol=5e-2 * y0.abs().max())


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_euler_solver_requires_multiple_points():
    with pytest.raises(AssertionError):
        euler_solver(lambda ti, y: y, torch.zeros(2), torch.tensor([0.0]))


# --------------------------------------------------------------------------- #
# coeff conversion idempotence + resize/zero/interpolate (fuzzed)
# --------------------------------------------------------------------------- #
@pytest.mark.oracle_smoke
@SETTINGS
@given(rows=st.integers(1, 8), cols=st.integers(1, 8), seed=st.integers(0, 10_000))
def test_to_complex_coeff_idempotent_on_complex(rows, cols, seed):
    c = torch.randn(
        rows, cols, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    assert to_complex_coeff(c) is c


@pytest.mark.oracle_smoke
@SETTINGS
@given(rows=st.integers(1, 8), cols=st.integers(1, 8), seed=st.integers(0, 10_000))
def test_to_real_coeff_idempotent_on_real(rows, cols, seed):
    r = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))
    assert to_real_coeff(r) is r


@pytest.mark.oracle_smoke
@SETTINGS
@given(cur=st.integers(2, 16), target=st.integers(2, 16), seed=st.integers(0, 10_000))
def test_resize_modes_int_target(cur, target, seed):
    x = torch.randn(
        2, cur, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    out = resize_modes(x, target)  # int target -> per-dim tuple
    assert out.shape == (2, target)


@pytest.mark.oracle_smoke
@SETTINGS
@given(n=st.integers(4, 16), cutoff=st.integers(0, 3), seed=st.integers(0, 10_000))
def test_zero_coeff_zeros_high_modes(n, cutoff, seed):
    x = torch.randn(
        2, n, dtype=torch.complex64, generator=torch.Generator().manual_seed(seed)
    )
    out = zero_coeff(x, WavenumberCutoff(cutoff))
    assert out.shape == x.shape
    assert torch.count_nonzero(out) <= torch.count_nonzero(x)


@pytest.mark.no_fuzz
@pytest.mark.oracle_analytic
def test_zero_coeff_per_axis_bands():
    x = torch.ones(2, 8, 8, dtype=torch.complex64)
    # |k| <= 1 keeps 3 bins of an axis, an exact count of 4 keeps 4
    mixed = zero_coeff(x, [WavenumberCutoff(1), ExactModeCount(4)])
    assert int(torch.count_nonzero(mixed)) == 2 * 3 * 4
    broadcast = zero_coeff(x, WavenumberCutoff(1))
    assert int(torch.count_nonzero(broadcast)) == 2 * 3 * 3


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_zero_coeff_rejects_a_bare_count():
    x = torch.ones(1, 8, dtype=torch.complex64)
    with pytest.raises(TypeError):
        zero_coeff(x, 4)
    with pytest.raises(TypeError):
        zero_coeff(x, [4])


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    slope=st.floats(-5.0, 5.0),
    intercept=st.floats(-5.0, 5.0),
    n=st.integers(3, 20),
    seed=st.integers(0, 10_000),
)
def test_interpolate_tensor_is_exact_on_linear_ramp(slope, intercept, n, seed):
    # MMS: a linear ramp y = slope*i + intercept is reproduced exactly by linear
    # interpolation at both integer and fractional indices.
    y = (slope * torch.arange(n, dtype=torch.float64) + intercept).reshape(1, n)
    g = torch.Generator().manual_seed(seed)
    idx = torch.rand(6, generator=g, dtype=torch.float64) * (n - 1)
    got = interpolate_tensor(y, idx, dim=1)
    exact = (slope * idx + intercept).reshape(1, -1)
    assert torch.allclose(got, exact, atol=1e-10)


@pytest.mark.oracle_analytic
@SETTINGS
@given(
    slope=st.floats(-5.0, 5.0),
    intercept=st.floats(-5.0, 5.0),
    n=st.integers(3, 20),
    seed=st.integers(0, 10_000),
)
def test_interpolate_tensor_integer_indices(slope, intercept, n, seed):
    # all-integer indices take the exact index_select branch; on a linear ramp
    # the selected values equal the closed form.
    y = (slope * torch.arange(n, dtype=torch.float64) + intercept).reshape(1, n)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (5,), generator=g).to(torch.float64)
    got = interpolate_tensor(y, idx, dim=1)
    exact = (slope * idx + intercept).reshape(1, -1)
    assert torch.allclose(got, exact, atol=1e-10)


# --------------------------------------------------------------------------- #
# Every conversion the package exports, not only the polar pair.
#
# coefficient-encodings binds *every* conversion to accept any number of mode
# axes and to preserve device and precision. Built from a list of pairs rather
# than one test per function, so a conversion added later is either covered here
# or fails the registry check.
# --------------------------------------------------------------------------- #
CONVERSIONS = (
    ("real/imag", to_real_coeff, to_complex_coeff),
    ("mag/angle", to_mag_angle, from_mag_angle),
)
CONVERSION_SHAPES = ((5, 9), (2, 8, 8), (2, 4, 4, 4))


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_conversion_registry_covers_every_exported_conversion():
    """The registry must track the package's exports, not drift from them."""
    import SpectralSVR

    exported = {
        name
        for name in dir(SpectralSVR)
        if name.startswith(("to_", "from_")) and name.endswith(("_coeff", "_angle"))
    }
    covered = {fn.__name__ for _, fwd, inv in CONVERSIONS for fn in (fwd, inv)}
    assert exported == covered, (
        f"registry and exports disagree: only exported {exported - covered}, "
        f"only in registry {covered - exported}"
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_every_conversion_accepts_any_number_of_mode_axes():
    generator = torch.Generator().manual_seed(21)
    for label, forward, inverse in CONVERSIONS:
        for shape in CONVERSION_SHAPES:
            c = torch.randn(*shape, dtype=torch.complex64, generator=generator)
            out = forward(c)
            assert out.shape == (*shape[:-1], shape[-1] * 2), f"{label} {shape}"
            assert torch.allclose(inverse(out), c, atol=1e-4), f"{label} {shape}"


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_every_conversion_preserves_precision():
    for label, forward, inverse in CONVERSIONS:
        for complex_dtype, real_dtype in (
            (torch.complex64, torch.float32),
            (torch.complex128, torch.float64),
        ):
            c = torch.randn(3, 4, dtype=complex_dtype)
            assert forward(c).dtype == real_dtype, label
            assert inverse(forward(c)).dtype == complex_dtype, label


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no accelerator available")
def test_every_conversion_preserves_device():
    for label, forward, inverse in CONVERSIONS:
        c = torch.randn(3, 4, dtype=torch.complex64).cuda()
        assert forward(c).device.type == "cuda", label
        assert inverse(forward(c)).device.type == "cuda", label
