"""Every scenario the main specs and in-flight deltas state is named by a test or exempt with a reason."""

import ast
import re
from pathlib import Path

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

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
    "A sampling of the wrong length is refused wherever it is resolved": IN_FLIGHT,
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


def _main_specs() -> list[Path]:
    return sorted((REPO / "openspec" / "specs").rglob("spec.md"))


def _deltas_in_flight() -> list[Path]:
    return sorted(
        path
        for path in (REPO / "openspec" / "changes").rglob("spec.md")
        if "archive" not in path.parts
    )


def _capabilities_stating_each_scenario(specs: list[Path]) -> dict[str, list[str]]:
    stated: dict[str, list[str]] = {}
    for spec in specs:
        for match in SCENARIO.finditer(spec.read_text()):
            stated.setdefault(match.group(1), []).append(spec.parent.name)
    return stated


def _stated() -> dict[str, list[str]]:
    return _capabilities_stating_each_scenario(_main_specs() + _deltas_in_flight())


def _scenarios_named_by_tests() -> set[str]:
    named: set[str] = set()
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
                named.update(
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                )
    return named


def uncovered_scenarios(
    stated: set[str], named: set[str], exempt: dict[str, str]
) -> list[str]:
    return sorted(stated - named - set(exempt))


def exemptions_no_spec_states(stated: set[str], exempt: dict[str, str]) -> list[str]:
    return sorted(name for name in exempt if name not in stated)


def exemptions_a_test_now_names(named: set[str], exempt: dict[str, str]) -> list[str]:
    return sorted(name for name in exempt if name in named)


def in_flight_exemptions_outliving_their_change(
    in_flight: set[str], exempt: dict[str, str]
) -> list[str]:
    return sorted(
        name
        for name, reason in exempt.items()
        if reason == IN_FLIGHT and name not in in_flight
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_invariant
def test_scenario_names_are_unique():
    collisions = {name: caps for name, caps in _stated().items() if len(set(caps)) > 1}
    assert not collisions, (
        f"scenario names reused across capabilities: {collisions}. The scenario "
        "marker keys on the name alone; rename one side."
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_every_stated_scenario_is_covered_or_exempt():
    uncovered = uncovered_scenarios(set(_stated()), _scenarios_named_by_tests(), EXEMPT)
    assert not uncovered, (
        f"{len(uncovered)} scenario(s) stated by a spec with no test naming them: "
        f"{uncovered}. Cover each with @pytest.mark.scenario(<name>) on the test "
        "that exercises it. EXEMPT is the last resort, not the first: an entry "
        "there is debt, and it needs a reason saying why no test can exist."
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_no_exemption_is_stale():
    gone = exemptions_no_spec_states(set(_stated()), EXEMPT)
    now_tested = exemptions_a_test_now_names(_scenarios_named_by_tests(), EXEMPT)
    assert not gone, (
        f"EXEMPT names scenarios no spec states any more: {gone}. Remove them."
    )
    assert not now_tested, (
        f"EXEMPT names scenarios that now have a test: {now_tested}. Remove the "
        "entry -- the list is debt to be worked down, and a stale entry hides "
        "coverage that already exists."
    )


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_every_in_flight_exemption_names_a_scenario_still_in_flight():
    in_flight = set(_capabilities_stating_each_scenario(_deltas_in_flight()))
    stranded = in_flight_exemptions_outliving_their_change(in_flight, EXEMPT)
    assert not stranded, (
        f"EXEMPT gives IN_FLIGHT for scenarios no change in flight states: "
        f"{stranded}. Their change was archived without a test naming them; "
        "cover each, or record the reason no test can exist."
    )


SCENARIOS = ["alpha", "beta", "gamma", "delta"]
CHANGES = ["first-change", "second-change"]


class ScenarioLedger(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.main: set[str] = set()
        self.in_flight: dict[str, set[str]] = {}
        self.named: set[str] = set()
        self.exempt: dict[str, str] = {}

    @property
    def scenarios_in_flight(self) -> set[str]:
        return set().union(*self.in_flight.values())

    @property
    def stated(self) -> set[str]:
        return self.main | self.scenarios_in_flight

    @initialize()
    def archive_a_change_whose_scenario_has_no_test(self):
        self.state_in_a_change("first-change", "alpha")
        self.exempt_while_in_flight("alpha")
        assert self.stranded() == []
        self.archive("first-change")
        assert self.stranded() == ["alpha"]
        self.name_in_a_test("alpha")
        assert exemptions_a_test_now_names(self.named, self.exempt) == ["alpha"]
        self.drop_exemption("alpha")
        assert uncovered_scenarios(self.stated, self.named, self.exempt) == []

    def stranded(self) -> list[str]:
        return in_flight_exemptions_outliving_their_change(
            self.scenarios_in_flight, self.exempt
        )

    @rule(change=st.sampled_from(CHANGES), scenario=st.sampled_from(SCENARIOS))
    def state_in_a_change(self, change, scenario):
        self.in_flight.setdefault(change, set()).add(scenario)

    @rule(change=st.sampled_from(CHANGES))
    def archive(self, change):
        self.main |= self.in_flight.pop(change, set())

    @rule(scenario=st.sampled_from(SCENARIOS))
    def name_in_a_test(self, scenario):
        self.named.add(scenario)

    @rule(scenario=st.sampled_from(SCENARIOS))
    def remove_the_test(self, scenario):
        self.named.discard(scenario)

    @rule(scenario=st.sampled_from(SCENARIOS))
    def exempt_while_in_flight(self, scenario):
        self.exempt[scenario] = IN_FLIGHT

    @rule(scenario=st.sampled_from(SCENARIOS))
    def exempt_as_a_decision(self, scenario):
        self.exempt[scenario] = DECISION

    @rule(scenario=st.sampled_from(SCENARIOS))
    def drop_exemption(self, scenario):
        self.exempt.pop(scenario, None)

    @invariant()
    def a_stated_scenario_is_reported_only_when_neither_tested_nor_exempt(self):
        reported = set(uncovered_scenarios(self.stated, self.named, self.exempt))
        for scenario in SCENARIOS:
            expected = (
                scenario in self.stated
                and scenario not in self.named
                and scenario not in self.exempt
            )
            assert (scenario in reported) == expected

    @invariant()
    def a_stale_exemption_is_reported(self):
        for scenario in self.exempt:
            assert (
                scenario in exemptions_no_spec_states(self.stated, self.exempt)
            ) == (scenario not in self.stated)
            assert (
                scenario in exemptions_a_test_now_names(self.named, self.exempt)
            ) == (scenario in self.named)

    @invariant()
    def an_in_flight_exemption_ends_with_its_change(self):
        stranded = self.stranded()
        for scenario, reason in self.exempt.items():
            assert (scenario in stranded) == (
                reason == IN_FLIGHT and scenario not in self.scenarios_in_flight
            )


ScenarioLedger.TestCase.settings = settings(max_examples=80, stateful_step_count=15)
TestScenarioLedger = pytest.mark.scenario("A scenario without a check is caught")(
    pytest.mark.scenario("The guard survives archiving")(
        pytest.mark.scenario("An exemption does not cover a later scenario")(
            pytest.mark.scenario("A scenario that cannot be code-tested is named")(
                pytest.mark.scenario("An in-flight exemption ends with its change")(
                    pytest.mark.oracle_invariant(ScenarioLedger.TestCase)
                )
            )
        )
    )
)
