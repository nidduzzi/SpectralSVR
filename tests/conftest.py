"""Testing-policy enforcement for the whole suite.

Every test must:

1. Be fuzzed with Hypothesis (``@given``) OR explicitly opt out with
   ``@pytest.mark.no_fuzz``.
2. Declare its oracle class with exactly one ``oracle_*`` marker -- what the
   test's expected values are anchored to:

   - ``oracle_independent``: an independent implementation (torch.fft, an
     inline naive solve, external reference data).
   - ``oracle_analytic``: a closed form / manufactured solution computed by
     hand-written math in the test (subsumes the old ``mms`` marker).
   - ``oracle_invariant``: a mathematical invariant a wrong implementation
     could not fake (linearity, conservation, convergence rate).
   - ``oracle_consistency``: self-referential -- output compared against other
     output of this codebase (round-trips, self-generated data).
   - ``oracle_smoke``: shape/dtype/no-crash/raises checks only.

A test that satisfies neither policy is reported as an error so the omission
is deliberate and visible rather than silent. ``tests/test_oracle_coverage.py``
additionally requires every non-exempt test module to carry at least one
independent/analytic/invariant anchor, so consistency tests can never be a
module's only line of defense.
"""

import matplotlib
import pytest

# Use a non-interactive backend so plotting tests run headless.
matplotlib.use("Agg")

ORACLE_MARKERS = {
    "oracle_independent": "expected values from an independent implementation",
    "oracle_analytic": "expected values from a closed form / manufactured solution",
    "oracle_invariant": "asserts a mathematical invariant (linearity, conservation, rate)",
    "oracle_consistency": "self-referential: compared against this codebase's own output",
    "oracle_smoke": "shape/dtype/no-crash/raises checks only",
}

# Strong anchors: at least one per module (see tests/test_oracle_coverage.py).
ANCHOR_MARKERS = ("oracle_independent", "oracle_analytic", "oracle_invariant")


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers", "no_fuzz: test is intentionally not fuzzed with Hypothesis"
    )
    config.addinivalue_line(
        "markers",
        'scenario(name): test covers the spec scenario of that name '
        "(see tests/test_scenario_coverage.py)",
    )
    for name, description in ORACLE_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def _is_hypothesis_test(item: pytest.Item) -> bool:
    if item.get_closest_marker("hypothesis") is not None:
        return True
    return getattr(getattr(item, "obj", None), "is_hypothesis_test", False)


def pytest_runtest_setup(item: pytest.Item):
    fuzzed = _is_hypothesis_test(item) or item.get_closest_marker("no_fuzz")
    if not fuzzed:
        raise Exception(
            f"{item.nodeid}: test is not fuzzed. Add a Hypothesis @given "
            "strategy or mark it @pytest.mark.no_fuzz to opt out explicitly."
        )
    oracles = [name for name in ORACLE_MARKERS if item.get_closest_marker(name)]
    if len(oracles) != 1:
        found = ", ".join(oracles) if oracles else "none"
        raise Exception(
            f"{item.nodeid}: test must declare exactly one oracle class "
            f"(found: {found}). Valid markers: {', '.join(ORACLE_MARKERS)}."
        )


@pytest.fixture(autouse=True)
def _isolate_torch_globals():
    """Snapshot and restore torch's global default dtype/device around each test.

    Some optional dependencies mutate torch globals on import (e.g. importing
    deepxde sets the default device to cuda), which would otherwise leak into
    every later test and cause device/dtype mismatches.
    """
    import torch

    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
