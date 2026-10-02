import dataclasses
import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from tests._prose import (
    BUDGETED_COLUMNS,
    DOCSTRING_SECTIONS,
    Budget,
    Prose,
    comment_lines,
    docstrings,
    is_in_shape,
    measure,
    over_budget,
    read_budget,
    slack,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET_FILE = ROOT / "tests" / "prose_budget.toml"
FILES = ["a.py", "b.py", "c.py"]
NOTHING = Prose(
    comments=0, internal_docstrings=0, public_docstrings=0, misshapen_docstrings=0
)


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_no_file_carries_more_prose_than_its_budget():
    assert over_budget(measure(ROOT), read_budget(BUDGET_FILE)) == []


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_no_budget_holds_slack():
    assert slack(measure(ROOT), read_budget(BUDGET_FILE)) == []


class BudgetLedger(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.measured = {
            file: dataclasses.replace(
                NOTHING, comments=2, internal_docstrings=1, misshapen_docstrings=1
            )
            for file in FILES[:2]
        }
        self.budget = {
            file: Budget(comments=2, internal_docstrings=1, misshapen_docstrings=1)
            for file in FILES[:2]
        }

    @initialize()
    def check_every_pinned_edge(self):
        self.add_prose("a.py", "comments")
        self.remove_prose("b.py", "comments")
        self.lower_the_budget_to_match("a.py")
        self.add_prose("c.py", "internal_docstrings")
        self.add_prose("c.py", "misshapen_docstrings")
        self.remove_prose("a.py", "misshapen_docstrings")

    @rule(file=st.sampled_from(FILES), column=st.sampled_from(BUDGETED_COLUMNS))
    def add_prose(self, file, column):
        prose = self.measured.get(file, NOTHING)
        self.measured[file] = dataclasses.replace(
            prose, **{column: getattr(prose, column) + 1}
        )

    @rule(file=st.sampled_from(FILES), column=st.sampled_from(BUDGETED_COLUMNS))
    def remove_prose(self, file, column):
        prose = self.measured.get(file, NOTHING)
        if getattr(prose, column) > 0:
            self.measured[file] = dataclasses.replace(
                prose, **{column: getattr(prose, column) - 1}
            )

    @rule(file=st.sampled_from(FILES))
    def lower_the_budget_to_match(self, file):
        prose = self.measured.get(file, NOTHING)
        allowance = self.budget.get(file, Budget())
        self.budget[file] = Budget(
            **{
                column: min(getattr(allowance, column), getattr(prose, column))
                for column in BUDGETED_COLUMNS
            }
        )

    @invariant()
    def each_file_is_held_to_its_own_budget(self):
        reported = over_budget(self.measured, self.budget)
        expected = [
            f"{file}: {column} {getattr(prose, column)} over its budget of "
            f"{getattr(self.budget.get(file, Budget()), column)}"
            for file, prose in self.measured.items()
            for column in BUDGETED_COLUMNS
            if getattr(prose, column) > getattr(self.budget.get(file, Budget()), column)
        ]
        assert sorted(reported) == sorted(expected)

    @invariant()
    def every_allowance_above_its_count_is_reported(self):
        expected = {
            (file, column)
            for file, allowance in self.budget.items()
            for column in BUDGETED_COLUMNS
            if getattr(allowance, column)
            > getattr(self.measured.get(file, NOTHING), column)
        }
        assert len(slack(self.measured, self.budget)) == len(expected)


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
    found = docstrings(source, None)

    assert comment_lines(source) == comments
    assert sum(d.lines for d in found if d.public) == public
    assert sum(d.lines for d in found if not d.public) == internal


NAMES = ["alpha", "beta", "_gamma", "__init__", "_delta"]


@pytest.mark.oracle_independent
@pytest.mark.scenario("Public and internal are told apart structurally")
@pytest.mark.scenario("The exemption is bounded")
@settings(max_examples=100)
@given(
    names=st.lists(st.sampled_from(NAMES), unique=True, min_size=1),
    exported=st.sets(st.sampled_from(NAMES)),
)
@example(names=["alpha"], exported=set())
@example(names=["alpha"], exported={"alpha"})
@example(names=["_gamma"], exported={"_gamma"})
def test_a_docstring_is_public_only_when_unprefixed_and_exported(names, exported):
    source = "\n".join(f'def {name}():\n    """{name}"""\n' for name in names)
    unprefixed = {
        name for name in names if not name.startswith("_") or name == "__init__"
    }

    library = {d.text: d.public for d in docstrings(source, frozenset(exported))}
    scripts = {d.text: d.public for d in docstrings(source, None)}

    assert library == {name: name in unprefixed and name in exported for name in names}
    assert scripts == {name: name in unprefixed for name in names}


ONE_SENTENCE = "Scale ``x.y`` by e.g. the factor ``1.5``."
TWO_SENTENCES = "Scale the input. It is then clipped."


@st.composite
def numpy_docstrings(draw):
    in_shape = True
    summary_sentences = draw(st.integers(1, 2))
    in_shape &= summary_sentences == 1
    parts = [ONE_SENTENCE if summary_sentences == 1 else TWO_SENTENCES]
    if draw(st.booleans()):
        parts.append("A free paragraph after the summary.")
        in_shape = False
    for name in draw(
        st.lists(
            st.sampled_from([*DOCSTRING_SECTIONS, "Notes"]), max_size=3, unique=True
        )
    ):
        in_shape &= name != "Notes"
        body = []
        for index in range(draw(st.integers(1, 2))):
            if name == "Examples":
                body.append(f">>> scale({index}.5)")
                continue
            two = draw(st.booleans())
            in_shape &= not two
            body.append(
                f"value_{index} : float\n    {TWO_SENTENCES if two else ONE_SENTENCE}"
            )
        parts.append(f"{name}\n{'-' * len(name)}\n" + "\n".join(body))
    return "\n\n".join(parts), in_shape


CONTRACT_WITH_MATHEMATICS = """Solve on a grid that resolves the manufactured band.

Parameters
----------
ns : int
    Grid modes, which must satisfy ``ns >= 6 * cutoff``.

Raises
------
ValueError
    Where ``ns < 6 * cutoff``, since the mask would discard the product's top modes."""

DERIVATION = """Solve on a grid that resolves the manufactured band.

The product of two fields at cutoff ``K`` reaches ``2 K`` while the mask keeps
``ns / 3``. The two agree when ``ns >= 6 K``, which ``ns = m`` satisfies."""


@pytest.mark.oracle_independent
@pytest.mark.scenario("A docstring outside the shape is refused")
@pytest.mark.scenario("A contract in the shape is accepted")
@settings(max_examples=200)
@given(case=numpy_docstrings())
@example(case=(ONE_SENTENCE, True))
@example(case=(TWO_SENTENCES, False))
@example(case=(CONTRACT_WITH_MATHEMATICS, True))
@example(case=(DERIVATION, False))
@example(
    case=(f"{ONE_SENTENCE}\n\nNotes\n-----\nA section outside the closed set.", False)
)
@example(case=("Parameters\n----------\nx : int\n    A value.", False))
@example(
    case=(f"{ONE_SENTENCE}\n\nExamples\n--------\n>>> scale(1.5)\n>>> scale(2.5)", True)
)
def test_the_shape_is_one_summary_sentence_and_one_sentence_per_entry(case):
    text, in_shape = case

    assert is_in_shape(text) == in_shape


SHAPED_BY = {
    "conforming": ('"""Return the value."""', 0),
    "ruff only, no trailing period": ('"""Return the value"""', 1),
    "ours only, two summary sentences": ('"""Return the value. It is cached."""', 1),
}


@pytest.mark.oracle_independent
@pytest.mark.scenario("A docstring outside the shape is refused")
@pytest.mark.scenario("The exemption covers length, not shape")
@settings(deadline=None, max_examples=6)
@given(kinds=st.lists(st.sampled_from(sorted(SHAPED_BY)), min_size=1, max_size=3))
@example(kinds=["conforming"])
@example(kinds=["ruff only, no trailing period"])
@example(kinds=["ours only, two summary sentences"])
def test_a_misshapen_docstring_is_one_ruff_flags_or_our_constraint_refuses(kinds):
    source = "\n\n".join(
        f"def value_{index}():\n    {SHAPED_BY[kind][0]}\n"
        for index, kind in enumerate(kinds)
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "tests").mkdir()
        (root / "pyproject.toml").write_text(
            '[tool.ruff.lint.pydocstyle]\nconvention = "numpy"\n'
        )
        (root / "tests" / "module.py").write_text(source)

        measured = measure(root)["tests/module.py"]

    assert measured.misshapen_docstrings == sum(SHAPED_BY[kind][1] for kind in kinds)


PACKAGE_NAMES = ["first", "second", "third"]


@pytest.mark.oracle_independent
@pytest.mark.scenario("Public and internal are told apart structurally")
@settings(deadline=None, max_examples=20)
@given(reexported=st.sets(st.sampled_from(PACKAGE_NAMES)))
@example(reexported=set())
@example(reexported={"first"})
@example(reexported=set(PACKAGE_NAMES))
def test_a_package_exports_what_its_init_reexports(reexported):
    module = "\n\n".join(
        f'def {name}():\n    """{name.title()}."""\n' for name in PACKAGE_NAMES
    )
    imports = "".join(
        f"from .module import {name} as {name}\n" for name in sorted(reexported)
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package = root / "SpectralSVR"
        package.mkdir()
        (package / "__init__.py").write_text(f'"""The package."""\n\n{imports}')
        (package / "module.py").write_text(module)

        measured = measure(root)

    assert measured["SpectralSVR/__init__.py"].public_docstrings == int(
        bool(reexported)
    )
    assert measured["SpectralSVR/__init__.py"].internal_docstrings == int(
        not reexported
    )
    assert measured["SpectralSVR/module.py"].public_docstrings == len(reexported)
    assert measured["SpectralSVR/module.py"].internal_docstrings == len(
        PACKAGE_NAMES
    ) - len(reexported)
