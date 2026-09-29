"""Scenario-level coverage guard.

Every scenario the specs state is either covered by a test naming it with
``@pytest.mark.scenario(...)`` or listed in ``EXEMPT`` with a reason. Reads the
main specs, which are the permanent contract, and the delta specs of changes
still in flight. Archived deltas are excluded: their requirements have already
been synced into the main specs, so counting both would double-count.

``EXEMPT`` is debt, not furniture. Entries come off it as scenarios gain tests;
nothing should be added without a reason that says why a test cannot exist.
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).parent
SCENARIO = re.compile(r"^#### Scenario:\s*(.+?)\s*$", re.M)

PREDATES = "predates scenario-level coverage; capability is anchored at module level"
DECISION = (
    "records a decision taken while writing the change; no run-time behaviour to test"
)
POLICY = "a rule about how tests are written; the guard cannot check it of itself"
IN_FLIGHT = "stated by a change still being implemented; covered as its tasks complete"

# Scenarios with no test naming them. Shrink this list; do not grow it.
EXEMPT: dict[str, str] = {
    "A case does not depend on the process default": IN_FLIGHT,
    # coefficient-encodings
    "Two mode axes convert": PREDATES,
    "One mode axis is unchanged": PREDATES,
    "Device is carried through": PREDATES,
    "Precision corresponds": PREDATES,
    "Round-trip through polar form": PREDATES,
    "Both encodings offer the same shape of API": PREDATES,
    "The flag's meaning is documented": PREDATES,
    "An even-count field stays real after padding": PREDATES,
    "An odd-count field is unaffected": PREDATES,
    "The represented function is unchanged": PREDATES,
    "Truncating to an even count keeps both halves": PREDATES,
    "Enlarging then reducing returns the original": PREDATES,
    "The limitation is stated where the caller meets it": PREDATES,
    "Documenting is not a substitute for a fix that exists": PREDATES,
    "Unstatable beats documented": PREDATES,
    "Band-limiting is no longer an instance of this": PREDATES,
    # coefficient-precision
    "Two precisions coexist": PREDATES,
    "Default is unchanged": PREDATES,
    "Double precision survives the model boundary": PREDATES,
    "A mismatch is not silent": PREDATES,
    "A widening mismatch proceeds but is reported": PREDATES,
    "Inputs a caller supplies are not governed by the basis's precision": PREDATES,
    "Exact round-trip at both precisions": PREDATES,
    "The device is carried through": PREDATES,
    "Non-contiguous input is accepted": PREDATES,
    "Target precision can be stated": PREDATES,
    "Cost is stated at the point of choice": PREDATES,
    # manufactured-solutions
    "Residual vanishes off the sample grid": PREDATES,
    "Real part taken before multiplying": PREDATES,
    "Same de-aliasing on both sides": PREDATES,
    "A wrong pair is caught": PREDATES,
    # numerical-anchor-tests
    "Fixed-array metric check": PREDATES,
    "rrse on near-constant target": PREDATES,
    "Time derivative of known field": PREDATES,
    "Truncation and padding values": PREDATES,
    "Mode zeroing values": PREDATES,
    "Matmul path on single-mode input": PREDATES,
    "Predictions match the KKT solve": PREDATES,
    "Forward matches closed-form antiderivative": PREDATES,
    "Inverse recovers known input": PREDATES,
    "Fuzzed analytic anchor": PREDATES,
    "Fixed-point anchor opts out explicitly": PREDATES,
    "Two integrators agree in a non-stiff regime": PREDATES,
    "Regime boundary is asserted, not assumed": PREDATES,
    "Convergence rate is asserted as an invariant": PREDATES,
    "Complex-eigenvalue weights are checked against the closed form": PREDATES,
    "The existing forcing test cannot be passed by a first-order method": PREDATES,
    "Continuous residual is the oracle": PREDATES,
    "A pair known to be correct reads clean": PREDATES,
    "Deliberate corruption is detected": PREDATES,
    "A convergence anchor runs at double precision": PREDATES,
    "Saturation is distinguished from order loss": PREDATES,
    # operator-comparison
    # pde-solver-protocol
    "Monolithic solvers remain callable": PREDATES,
    "Semilinear solver requires the split form": PREDATES,
    "Conversion is result-preserving": PREDATES,
    "Complex spectral state is supported": PREDATES,
    "No conversion in the other direction": PREDATES,
    "Supplied solver is the one that runs": PREDATES,
    "Monolithic solver via conversion": PREDATES,
    "Wrong interface is rejected": PREDATES,
    "Default generation is unchanged": PREDATES,
    "Regime is documented": PREDATES,
    "Time grid is passed through unchanged": PREDATES,
    # semilinear-time-integration
    "Convergence rate under time-dependent forcing": PREDATES,
    "Time-independent nonlinear term is unaffected": PREDATES,
    "Complex eigenvalues match the closed form": PREDATES,
    "Real eigenvalues are unchanged": PREDATES,
    "Branch is decided by imaginary content, not dtype": PREDATES,
    "Non-uniform grid is rejected": PREDATES,
    "Uniform grid from linspace is accepted": PREDATES,
    "Docstrings match behaviour": PREDATES,
    # spectral-band-selection
    "The kept set is evident at the call site": PREDATES,
    "A bare count is not accepted": PREDATES,
    "A symmetric band about the zero mode": PREDATES,
    "The whole grid is a statable band": PREDATES,
    "An unsatisfiable band cannot be stated": PREDATES,
    "A guaranteeing band is accepted": PREDATES,
    "A band without the guarantee is refused where it is passed": PREDATES,
    "Band-limiting itself does not require the guarantee": PREDATES,
    "An exact count is honoured": PREDATES,
    "The absent guarantee is stated": PREDATES,
    "Choosing it is deliberate": PREDATES,
    # test-oracle-convention
    "Unmarked test fails at setup": PREDATES,
    "Multiply-marked test fails at setup": PREDATES,
    "Correctly marked test runs": PREDATES,
    "Former MMS test": PREDATES,
    "MMS enforcement removed": PREDATES,
    "Module with only consistency and smoke tests": PREDATES,
    "Exempt module": PREDATES,
    "Round-trip test classification": PREDATES,
    "External-reference test classification": PREDATES,
    "The operators exemption is re-examined": DECISION,
    "Withdrawal never leaves a failing guard": DECISION,
    "The anchor is preferred to the record": DECISION,
    "Scenarios over an input range are fuzzed": POLICY,
}


def _spec_files() -> list[Path]:
    main = sorted((REPO / "openspec" / "specs").rglob("spec.md"))
    active = sorted(
        path
        for path in (REPO / "openspec" / "changes").rglob("spec.md")
        if "archive" not in path.parts
    )
    return main + active


def _stated() -> dict[str, list[str]]:
    """Scenario name -> the capabilities stating it."""
    stated: dict[str, list[str]] = {}
    for spec in _spec_files():
        for match in SCENARIO.finditer(spec.read_text()):
            stated.setdefault(match.group(1), []).append(spec.parent.name)
    return stated


def _covered() -> set[str]:
    """Scenario names named by a @pytest.mark.scenario(...) anywhere in tests/."""
    covered: set[str] = set()
    for module in TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in module.parts:
            continue
        for node in ast.walk(ast.parse(module.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "scenario"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "mark"
            ):
                covered.update(
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                )
    return covered


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_scenario_names_are_unique():
    collisions = {name: caps for name, caps in _stated().items() if len(set(caps)) > 1}
    assert not collisions, (
        f"scenario names reused across capabilities: {collisions}. The scenario "
        "marker keys on the name alone; rename one side."
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
@pytest.mark.scenario("A scenario without a check is caught")
@pytest.mark.scenario("The guard survives archiving")
def test_every_stated_scenario_is_covered_or_exempt():
    uncovered = sorted(set(_stated()) - _covered() - set(EXEMPT))
    assert not uncovered, (
        f"{len(uncovered)} scenario(s) stated by a spec with no test naming them: "
        f"{uncovered}. Cover each with @pytest.mark.scenario(<name>) on the test "
        "that exercises it. EXEMPT is the last resort, not the first: an entry "
        "there is debt, and it needs a reason saying why no test can exist."
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
@pytest.mark.scenario("An exemption does not cover a later scenario")
@pytest.mark.scenario("A scenario that cannot be code-tested is named")
def test_no_exemption_is_stale():
    stated, covered = set(_stated()), _covered()
    gone = sorted(name for name in EXEMPT if name not in stated)
    now_tested = sorted(name for name in EXEMPT if name in covered)
    assert not gone, (
        f"EXEMPT names scenarios no spec states any more: {gone}. Remove them."
    )
    assert not now_tested, (
        f"EXEMPT names scenarios that now have a test: {now_tested}. Remove the "
        "entry -- the list is debt to be worked down, and a stale entry hides "
        "coverage that already exists."
    )
