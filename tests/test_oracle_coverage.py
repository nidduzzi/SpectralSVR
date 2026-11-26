"""Per-module anchor coverage guard.

Every non-exempt test module must contain at least one test anchored to
something outside this codebase's own output: an independent implementation,
a closed form, or a mathematical invariant (the ``ANCHOR_MARKERS`` in
conftest). Consistency (round-trip) and smoke tests alone cannot catch a bug
that affects both sides of a comparison -- see the Fourier rescale incident.
"""

import ast
import itertools
from pathlib import Path

import pytest

from .conftest import ANCHOR_MARKERS, ORACLE_MARKERS

TESTS_DIR = Path(__file__).parent

# Modules allowed to carry no independent/analytic/invariant anchor. Keep this
# list short and justified: pure plumbing whose failure modes are structural
# (wrong shape/exception), not numerical.
EXEMPT = {
    "basis/test_config.py",  # SpectralConfig plumbing: immutability/validation only
    "model/test_fnn.py",  # NN trainer has no closed form; overfit + serialization only
    "test_oracle_coverage.py",  # this guard
}


def _test_modules() -> list[Path]:
    return [p for p in TESTS_DIR.rglob("test_*.py") if "__pycache__" not in p.parts]


def _marker_names(module: Path) -> set[str]:
    """All pytest marker names used in a module (decorators + pytestmark)."""
    tree = ast.parse(module.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        # @pytest.mark.<name> (possibly called) on functions/classes
        if isinstance(node, ast.Attribute):
            parent = node.value
            if (
                isinstance(parent, ast.Attribute)
                and parent.attr == "mark"
                and isinstance(parent.value, ast.Name)
                and parent.value.id == "pytest"
            ):
                names.add(node.attr)
    return names


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_every_module_has_an_anchor():
    missing = []
    for module in _test_modules():
        rel = str(module.relative_to(TESTS_DIR))
        if rel in EXEMPT:
            continue
        used = _marker_names(module)
        if not used.intersection(ANCHOR_MARKERS):
            missing.append(rel)
    assert not missing, (
        "test modules without any independent/analytic/invariant anchor "
        f"(only consistency/smoke coverage): {sorted(missing)}. Add an anchor "
        "test or add the module to EXEMPT with a justification."
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_exempt_list_is_current():
    # exempt entries must exist, and must actually lack anchors (otherwise the
    # exemption is stale and should be removed)
    for rel in EXEMPT - {"test_oracle_coverage.py"}:
        module = TESTS_DIR / rel
        assert module.exists(), f"EXEMPT entry {rel} does not exist"
        used = _marker_names(module)
        assert not used.intersection(ANCHOR_MARKERS), (
            f"EXEMPT entry {rel} now has an anchor test; remove the exemption"
        )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_mms_markers_are_retired():
    offenders = []
    for module in _test_modules():
        used = _marker_names(module)
        if used.intersection({"mms", "no_mms"}):
            offenders.append(str(module.relative_to(TESTS_DIR)))
    assert not offenders, f"retired mms/no_mms markers still present: {offenders}"


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_oracle_marker_names_match_conftest():
    # guard against typo'd oracle_* markers that setup enforcement would treat
    # as "no oracle declared"
    unknown = []
    for module in _test_modules():
        unknown.extend(
            f"{module.name}:{name}"
            for name in _marker_names(module)
            if name.startswith("oracle_") and name not in ORACLE_MARKERS
        )
    assert not unknown, f"unknown oracle markers: {unknown}"


class _FakeItem:
    """Minimal pytest.Item stand-in for exercising the setup enforcement."""

    def __init__(self, markers: set[str]):
        self._markers = markers
        self.nodeid = "fake::test"
        self.obj = None

    def get_closest_marker(self, name: str):
        return object() if name in self._markers else None


ORACLE_MARKER_SETS = [
    (set(), "found: none"),
    *(({one}, None) for one in ORACLE_MARKERS),
    *(
        (set(pair), ", ".join(pair))
        for pair in itertools.combinations(ORACLE_MARKERS, 2)
    ),
]


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
@pytest.mark.parametrize(
    "oracles, refusal",
    ORACLE_MARKER_SETS,
    ids=["+".join(sorted(oracles)) or "none" for oracles, _ in ORACLE_MARKER_SETS],
)
def test_setup_enforcement_judges_every_marker_set(oracles, refusal):
    from .conftest import pytest_runtest_setup

    item = _FakeItem({"no_fuzz", *oracles})
    if refusal is None:
        pytest_runtest_setup(item)
    else:
        with pytest.raises(Exception, match=f"exactly one oracle class.*{refusal}"):
            pytest_runtest_setup(item)
