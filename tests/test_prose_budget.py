import dataclasses
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from tests._prose import (
    Budget,
    Prose,
    comment_lines,
    docstring_lines,
    measure,
    over_budget,
    read_budget,
    slack,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET_FILE = ROOT / "tests" / "prose_budget.toml"
FILES = ["a.py", "b.py", "c.py"]


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
@pytest.mark.scenario("A file over its budget fails")
@pytest.mark.scenario("A new file starts at zero")
def test_no_file_carries_more_prose_than_its_budget():
    assert over_budget(measure(ROOT), read_budget(BUDGET_FILE)) == []


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
@pytest.mark.scenario("Slack is reclaimed, not banked")
def test_no_budget_holds_slack():
    assert slack(measure(ROOT), read_budget(BUDGET_FILE)) == []


class BudgetLedger(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.measured = {file: Prose(2, 1, 0) for file in FILES[:2]}
        self.budget = {file: Budget(2, 1) for file in FILES[:2]}

    @initialize()
    def check_every_pinned_edge(self):
        self.add_prose("a.py", "comments")
        self.remove_prose("b.py", "comments")
        self.lower_the_budget_to_match("a.py")
        self.add_prose("c.py", "internal_docstrings")

    @rule(
        file=st.sampled_from(FILES),
        column=st.sampled_from(["comments", "internal_docstrings"]),
    )
    def add_prose(self, file, column):
        prose = self.measured.get(file, Prose(0, 0, 0))
        self.measured[file] = dataclasses.replace(
            prose, **{column: getattr(prose, column) + 1}
        )

    @rule(
        file=st.sampled_from(FILES),
        column=st.sampled_from(["comments", "internal_docstrings"]),
    )
    def remove_prose(self, file, column):
        prose = self.measured.get(file)
        if prose is not None and getattr(prose, column) > 0:
            self.measured[file] = dataclasses.replace(
                prose, **{column: getattr(prose, column) - 1}
            )

    @rule(file=st.sampled_from(FILES))
    def lower_the_budget_to_match(self, file):
        prose = self.measured.get(file, Prose(0, 0, 0))
        allowance = self.budget.get(file, Budget())
        self.budget[file] = Budget(
            min(allowance.comments, prose.comments),
            min(allowance.internal_docstrings, prose.internal_docstrings),
        )

    @invariant()
    def each_file_is_held_to_its_own_budget(self):
        expected_over = {
            (file, column)
            for file, prose in self.measured.items()
            for column in ("comments", "internal_docstrings")
            if getattr(prose, column) > getattr(self.budget.get(file, Budget()), column)
        }
        reported = over_budget(self.measured, self.budget)
        assert len(reported) == len(expected_over)
        for file, column in expected_over:
            allowance = getattr(self.budget.get(file, Budget()), column)
            count = getattr(self.measured[file], column)
            assert (
                f"{file}: {column} {count} over its budget of {allowance}" in reported
            )

    @invariant()
    def every_allowance_above_its_count_is_reported(self):
        expected_slack = {
            (file, column)
            for file, allowance in self.budget.items()
            for column in ("comments", "internal_docstrings")
            if getattr(allowance, column)
            > getattr(self.measured.get(file, Prose(0, 0, 0)), column)
        }
        assert len(slack(self.measured, self.budget)) == len(expected_slack)


BudgetLedger.TestCase.settings = settings(max_examples=60, stateful_step_count=12)
TestBudgetLedger = pytest.mark.scenario("The budget is per file, not a pooled total")(
    pytest.mark.scenario("A file over its budget fails")(
        pytest.mark.scenario("A new file starts at zero")(
            pytest.mark.scenario("Slack is reclaimed, not banked")(
                pytest.mark.oracle_independent(BudgetLedger.TestCase)
            )
        )
    )
)


PROSE_COMMENT = "# narration"
DIRECTIVES = [
    "# noqa: E501",
    "# type: ignore[misc]",
    "# pragma: no cover",
    "# ruff: noqa",
    "# mypy: ignore-errors",
    "# pyright: ignore",
    "# fmt: off",
    "# isort: skip",
]


@st.composite
def constructed_modules(draw):
    pieces, comments, public, internal = [], 0, 0, 0
    for index in range(draw(st.integers(0, 6))):
        kind = draw(
            st.sampled_from(
                [
                    "comment",
                    "directive",
                    "hash_string",
                    "quoted_comment",
                    "function",
                    "class",
                ]
            )
        )
        if kind == "comment":
            pieces.append(PROSE_COMMENT)
            comments += 1
        elif kind == "directive":
            pieces.append(f"x_{index} = 1  {draw(st.sampled_from(DIRECTIVES))}")
        elif kind == "hash_string":
            pieces.append(f'x_{index} = "# not a comment"')
        elif kind == "quoted_comment":
            pieces.append('# a """ inside a comment')
            comments += 1
        else:
            hidden = draw(st.booleans())
            lines = draw(st.integers(1, 4))
            name = f"{'_' if hidden else ''}name_{index}"
            body = "\n    ".join(f"line {line}" for line in range(lines))
            keyword = "def" if kind == "function" else "class"
            signature = f"{name}()" if keyword == "def" else name
            pieces.append(f'{keyword} {signature}:\n    """{body}"""\n')
            internal, public = (
                (internal + lines, public) if hidden else (internal, public + lines)
            )
    return "\n".join(pieces) + "\n", comments, public, internal


@pytest.mark.oracle_independent
@pytest.mark.scenario("Machine-directed comments are not prose")
@settings(max_examples=200)
@given(module=constructed_modules())
@example(module=("", 0, 0, 0))
@example(module=('x = "# not a comment"\n', 0, 0, 0))
@example(module=('# a """ inside a comment\n', 1, 0, 0))
@example(module=("\n".join(f"x = 1  {d}" for d in DIRECTIVES) + "\n", 0, 0, 0))
def test_constructed_modules_count_as_built(module):
    source, comments, public, internal = module

    assert comment_lines(source) == comments
    assert docstring_lines(source, None) == (public, internal)


@pytest.mark.oracle_independent
@pytest.mark.scenario("Public and internal are told apart structurally")
@pytest.mark.scenario("The exemption is bounded")
@settings(max_examples=100)
@given(
    names=st.lists(
        st.sampled_from(["alpha", "beta", "_gamma", "__init__", "_delta"]),
        unique=True,
        min_size=1,
    ),
    exported=st.sets(
        st.sampled_from(["alpha", "beta", "_gamma", "__init__", "_delta"])
    ),
)
@example(names=["alpha"], exported=set())
@example(names=["alpha"], exported={"alpha"})
@example(names=["_gamma"], exported={"_gamma"})
def test_a_docstring_is_public_only_when_unprefixed_and_exported(names, exported):
    source = "\n".join(f'def {name}():\n    """{name}"""\n' for name in names)
    public_names = [
        name for name in names if not name.startswith("_") or name == "__init__"
    ]

    library_public, library_internal = docstring_lines(source, frozenset(exported))
    script_public, script_internal = docstring_lines(source, None)

    exported_public = [name for name in public_names if name in exported]
    assert library_public == len(exported_public)
    assert library_internal == len(names) - len(exported_public)
    assert script_public == len(public_names)
    assert script_internal == len(names) - len(public_names)
