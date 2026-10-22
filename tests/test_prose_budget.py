import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from tests._prose import (
    BUDGETED_COLUMNS,
    Budget,
    BudgetFileError,
    Prose,
    comment_lines,
    docstring_violation_rows,
    docstrings,
    is_in_shape,
    library_exports,
    measure,
    over_budget,
    read_budget,
    ruff_docstring_report,
    sentence_count,
    slack,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET_FILE = ROOT / "tests" / "prose_budget.toml"
NOTHING = Prose(
    comments=0, internal_docstrings=0, public_docstrings=0, misshapen_docstrings=0
)
SPEC_SECTIONS = (
    "Parameters",
    "Other Parameters",
    "Receives",
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Attributes",
    "Methods",
    "See Also",
    "Examples",
)
FREE_PROSE_SECTIONS = ("Notes", "Warnings", "References")
STATED_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "cf.",
    "vs.",
    "et al.",
    "approx.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Fig.",
    "Eq.",
    "No.",
)
DOCUMENTED_DIRECTIVES = (
    "# noqa",
    "# noqa: E501",
    "# NOQA",
    "# type: ignore",
    "# type: ignore[misc]",
    "# pragma: no cover",
    "# ruff: noqa",
    "# mypy: ignore-errors",
    "# pyright: ignore",
    "# pyright: basic",
    "# pylint: disable=invalid-name",
    "# fmt: off",
    "# fmt: skip",
    "# isort: skip",
)
FILE_HEADERS = ("#!/usr/bin/env python", "# -*- coding: utf-8 -*-")


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_no_file_carries_more_prose_than_its_budget():
    breaches = over_budget(measure(ROOT), read_budget(BUDGET_FILE))
    assert breaches == [], [
        f"{b.file}: {b.column} {b.count} over its budget of {b.allowance}"
        for b in breaches
    ]


@pytest.mark.no_fuzz
@pytest.mark.oracle_consistency
def test_no_budget_holds_slack():
    breaches = slack(measure(ROOT), read_budget(BUDGET_FILE))
    assert breaches == [], [
        f"{b.file}: {b.column} budget {b.allowance} is above the {b.count} it carries; lower it"
        for b in breaches
    ]


@pytest.mark.no_fuzz
@pytest.mark.oracle_independent
def test_every_ruff_docstring_finding_lands_in_a_counted_docstring():
    flagged = docstring_violation_rows(ruff_docstring_report(ROOT), ROOT)
    stranded = [
        (file, row)
        for file, rows in flagged.items()
        for row in rows
        if not any(
            d.first_line <= row <= d.last_line
            for d in docstrings((ROOT / file).read_text(), None)
        )
    ]
    assert stranded == []


FILES = ["a.py", "b.py", "c.py"]


class BudgetLedger(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        held = dataclasses.replace(
            NOTHING, comments=2, internal_docstrings=1, misshapen_docstrings=1
        )
        self.measured = {file: held for file in FILES[:2]}
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
    def a_file_is_reported_over_exactly_where_it_carries_more_than_it_may(self):
        reported = {(b.file, b.column) for b in over_budget(self.measured, self.budget)}
        for file in FILES:
            for column in BUDGETED_COLUMNS:
                carried = getattr(self.measured.get(file, NOTHING), column)
                allowed = getattr(self.budget.get(file, Budget()), column)
                assert ((file, column) in reported) == (
                    file in self.measured and carried > allowed
                )

    @invariant()
    def an_allowance_is_reported_as_slack_exactly_where_it_exceeds_the_count(self):
        reported = {(b.file, b.column) for b in slack(self.measured, self.budget)}
        for file in FILES:
            for column in BUDGETED_COLUMNS:
                carried = getattr(self.measured.get(file, NOTHING), column)
                allowed = getattr(self.budget.get(file, Budget()), column)
                assert ((file, column) in reported) == (
                    file in self.budget and allowed > carried
                )


BudgetLedger.TestCase.settings = settings(max_examples=60, stateful_step_count=12)
TestBudgetLedger = pytest.mark.scenario("The budget is per file, not a pooled total")(
    pytest.mark.scenario("A file over its budget fails")(
        pytest.mark.scenario("A new file starts at zero")(
            pytest.mark.scenario("Slack is reclaimed, not banked")(
                pytest.mark.oracle_invariant(BudgetLedger.TestCase)
            )
        )
    )
)


@st.composite
def commented_sources(draw):
    header = list(FILE_HEADERS) if draw(st.booleans()) else []
    pieces, prose = [*header], 0
    for index in range(draw(st.integers(0, 6))):
        kind = draw(
            st.sampled_from(["comment", "directive", "hash_string", "quoted_comment"])
        )
        if kind == "comment":
            pieces.append("# narration")
            prose += 1
        elif kind == "directive":
            pieces.append(
                f"x_{index} = 1  {draw(st.sampled_from(DOCUMENTED_DIRECTIVES))}"
            )
        elif kind == "hash_string":
            pieces.append(f'x_{index} = "# not a comment"')
        else:
            pieces.append('# a """ inside a comment')
            prose += 1
    return "\n".join(pieces) + "\n", prose


@pytest.mark.oracle_independent
@pytest.mark.scenario("Machine-directed comments are not prose")
@settings(max_examples=200)
@given(case=commented_sources())
@example(case=("", 0))
@example(case=("\n".join(FILE_HEADERS) + "\n", 0))
@example(case=("\n".join(f"x = 1  {d}" for d in DOCUMENTED_DIRECTIVES) + "\n", 0))
@example(case=('x = "# not a comment"\n', 0))
@example(case=('# a """ inside a comment\n', 1))
def test_only_prose_comments_are_counted(case):
    source, prose = case

    assert comment_lines(source) == prose


@st.composite
def counted_sentences(draw):
    sentences = []
    for index in range(draw(st.integers(0, 4))):
        insert = draw(
            st.sampled_from(
                [
                    "",
                    "``a.b``",
                    ":func:`m.n`",
                    ":math:`x? y`",
                    "1.5",
                    "wait... then",
                    *STATED_ABBREVIATIONS,
                ]
            )
        )
        sentences.append(
            f"Clause {index} {insert} holds{draw(st.sampled_from(['.', '?', '!']))}"
        )
    if draw(st.booleans()):
        sentences.append("and a trailing fragment")
    return " ".join(sentences), len(sentences)


@pytest.mark.oracle_analytic
@pytest.mark.scenario("A docstring outside the shape is refused")
@settings(max_examples=300)
@given(case=counted_sentences())
@example(case=("", 0))
@example(case=("Do x", 1))
@example(case=("Do x. And y", 2))
@example(case=("Is it? Yes!", 2))
@example(case=("Wait... then go.", 1))
@example(case=("Ask Dr. Smith.", 1))
@example(case=("Scale ``x.y`` by e.g. the factor ``1.5``.", 1))
@example(case=("Bound :math:`x? y` holds.", 1))
def test_sentences_are_counted_as_written(case):
    text, count = case

    assert sentence_count(text) == count


ONE_SENTENCE = "Scale ``x.y`` by e.g. the factor ``1.5``."
TWO_SENTENCES = "Scale the input. It is then clipped."


@st.composite
def numpy_docstrings(draw):
    in_shape = draw(st.booleans())
    parts = [ONE_SENTENCE if in_shape else TWO_SENTENCES]
    if draw(st.booleans()):
        parts.append("A free paragraph after the summary.")
        in_shape = False
    names = draw(
        st.lists(
            st.sampled_from([*SPEC_SECTIONS, *FREE_PROSE_SECTIONS]),
            max_size=3,
            unique=True,
        )
    )
    for name in names:
        in_shape = in_shape and name in SPEC_SECTIONS
        underline = "-" * draw(st.integers(1, len(name) + 2))
        if name == "Examples":
            prose = draw(st.booleans())
            in_shape = in_shape and not prose
            body = (
                "An introduction.\n>>> scale(1)"
                if prose
                else ">>> scale(1)\n1\n>>> scale(2)\n... # continued"
            )
        else:
            two = draw(st.booleans())
            in_shape = in_shape and not two
            description = TWO_SENTENCES if two else ONE_SENTENCE
            body = (
                f"value : float\n    {description}"
                if name != "See Also"
                else f"scale : {description}"
            )
        parts.append(f"{name}\n{underline}\n{body}")
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

DERIVATION_IN_EXAMPLES = (
    "Return the band.\n\nExamples\n--------\n"
    "The product reaches 2K. The mask keeps ns/3. So ns >= 6K.\n>>> band(12)"
)


@pytest.mark.oracle_independent
@pytest.mark.scenario("A docstring outside the shape is refused")
@pytest.mark.scenario("A contract in the shape is accepted")
@settings(max_examples=300)
@given(case=numpy_docstrings())
@example(case=(ONE_SENTENCE, True))
@example(case=(TWO_SENTENCES, False))
@example(case=("Do x. And y", False))
@example(case=(CONTRACT_WITH_MATHEMATICS, True))
@example(case=(DERIVATION, False))
@example(case=(DERIVATION_IN_EXAMPLES, False))
@example(case=(f"{ONE_SENTENCE}\n\nSee Also\n--------\nscale : {TWO_SENTENCES}", False))
@example(case=(f"{ONE_SENTENCE}\n\nSee Also\n--------\nscale : {ONE_SENTENCE}", True))
@example(
    case=(f"{ONE_SENTENCE}\n\nNotes\n-----\nA section outside the closed set.", False)
)
@example(case=("Parameters\n----------\nx : int\n    A value.", False))
@example(
    case=(
        f"{ONE_SENTENCE}\n\nExamples\n--------\n>>> scale(1.5)\n2.25\n\nNotes\n-----\nProse.",
        False,
    )
)
def test_the_shape_is_one_summary_sentence_and_one_sentence_per_entry(case):
    text, in_shape = case

    assert is_in_shape(text) == in_shape


@st.composite
def nested_definitions(draw):
    outer, inner, method = (draw(st.booleans()) for _ in range(3))
    exported = draw(st.booleans())
    guard = draw(st.sampled_from(["", "if True:", "try:", "with open(__file__):"]))
    outer_name = "Outer" if outer else "_Outer"
    body = (
        f'class {outer_name}:\n    """Outer."""\n\n'
        f'    class {"Inner" if inner else "_Inner"}:\n        """Inner."""\n\n'
        f'        def {"method" if method else "_method"}(self):\n            """Method."""\n\n'
        f'            def nested():\n                """Nested."""\n'
    )
    if guard:
        handler = "\nexcept OSError:\n    pass\n" if guard == "try:" else "\n"
        body = (
            guard
            + "\n"
            + "\n".join("    " + line for line in body.splitlines())
            + handler
        )
    outer_public = outer and exported
    inner_public = outer_public and inner
    expected = {
        "Outer.": outer_public,
        "Inner.": inner_public,
        "Method.": inner_public and method,
        "Nested.": False,
    }
    names = frozenset({outer_name} if exported else set())
    return body, names, expected


@pytest.mark.oracle_independent
@pytest.mark.scenario("Public and internal are told apart structurally")
@pytest.mark.scenario("The exemption is bounded")
@settings(max_examples=150)
@given(case=nested_definitions())
@example(
    case=('class Outer:\n    """Outer."""\n', frozenset({"Outer"}), {"Outer.": True})
)
@example(
    case=(
        'if True:\n    def hidden():\n        """Hidden. Two."""\n',
        frozenset(),
        {"Hidden. Two.": False},
    )
)
def test_visibility_follows_names_exports_and_nesting(case):
    source, exported, expected = case

    found = {d.text: d.public for d in docstrings(source, exported)}

    assert found == expected


STYLES = ["absent", "relative", "absolute", "conditional", "star"]
MODULE_NAMES = ["first", "second", "third", "fourth"]


def names_python_binds_from_the_module(root: Path) -> set[str]:
    probe = (
        "import json, SpectralSVR\n"
        "assert SpectralSVR.__file__.startswith(%r)\n"
        "print(json.dumps(sorted(n for n in dir(SpectralSVR)"
        " if getattr(getattr(SpectralSVR, n), '__module__', '') == 'SpectralSVR.module')))"
    ) % str(root)
    output = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(json.loads(output))


@pytest.mark.oracle_independent
@pytest.mark.scenario("Public and internal are told apart structurally")
@settings(deadline=None, max_examples=8)
@given(
    styles=st.lists(
        st.sampled_from(STYLES), min_size=len(MODULE_NAMES), max_size=len(MODULE_NAMES)
    )
)
@example(styles=["absent"] * 4)
@example(styles=["relative", "absolute", "conditional", "star"])
def test_the_export_map_agrees_with_what_python_binds(styles):
    module = "\n\n".join(
        f'def {name}():\n    """{name.title()}."""\n' for name in MODULE_NAMES
    )
    starred = [
        name
        for name, style in zip(MODULE_NAMES, styles, strict=True)
        if style == "star"
    ]
    module += f"\n\n__all__ = {starred!r}\n"
    lines = ['"""The package."""']
    for name, style in zip(MODULE_NAMES, styles, strict=True):
        if style == "relative":
            lines.append(f"from .module import {name} as {name}")
        elif style == "absolute":
            lines.append(f"from SpectralSVR.module import {name} as {name}")
        elif style == "conditional":
            lines.append(f"if True:\n    from .module import {name} as {name}")
    if starred:
        lines.append("from .module import *")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        package = root / "SpectralSVR"
        package.mkdir()
        (package / "__init__.py").write_text("\n".join(lines) + "\n")
        (package / "module.py").write_text(module)

        exported = library_exports(root)
        bound = names_python_binds_from_the_module(root)

    assert set(exported.get(package / "module.py", frozenset())) == bound
    assert bool(exported.get(package / "__init__.py")) == bool(bound)


SHAPED_BY = {
    "conforming": ('"""Return the value."""', 0),
    "ruff only, no trailing period": ('"""Return the value"""', 1),
    "ruff only, short underline": (
        '"""Return the value.\n\n    Parameters\n    ---\n    x : int\n        A value.\n    """',
        1,
    ),
    "ours only, two summary sentences": ('"""Return the value. It is cached."""', 1),
}


@pytest.mark.oracle_independent
@pytest.mark.scenario("A docstring outside the shape is refused")
@pytest.mark.scenario("The exemption covers length, not shape")
@settings(deadline=None, max_examples=6)
@given(kinds=st.lists(st.sampled_from(sorted(SHAPED_BY)), min_size=1, max_size=3))
@example(kinds=["conforming"])
@example(kinds=["ruff only, no trailing period"])
@example(kinds=["ruff only, short underline"])
@example(kinds=["ours only, two summary sentences"])
def test_a_misshapen_docstring_is_one_ruff_flags_or_our_constraint_refuses(kinds):
    source = "\n\n".join(
        f"def value_{index}(x):\n    {SHAPED_BY[kind][0]}\n"
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


@pytest.mark.oracle_independent
@settings(max_examples=60)
@given(
    codes=st.lists(
        st.sampled_from(["D400", "D409", "E501", "invalid-syntax", None]), max_size=6
    )
)
@example(codes=[None])
@example(codes=["invalid-syntax", "D400"])
def test_only_docstring_rule_findings_are_joined(codes):
    root = Path("/repository")
    report = [
        {
            "code": code,
            "filename": "/repository/tests/module.py",
            "location": {"row": row},
        }
        for row, code in enumerate(codes, start=1)
    ]

    rows = docstring_violation_rows(report, root).get("tests/module.py", set())

    assert rows == {
        row for row, code in enumerate(codes, start=1) if code and code.startswith("D")
    }


@pytest.mark.oracle_independent
@settings(max_examples=80)
@given(
    columns=st.dictionaries(
        st.sampled_from([*BUDGETED_COLUMNS, "bogus"]),
        st.one_of(st.integers(-2, 3), st.booleans()),
        max_size=3,
    )
)
@example(columns={"comments": 1})
@example(columns={"bogus": 1})
@example(columns={"comments": -1})
@example(columns={"comments": True})
def test_a_budget_file_states_only_known_non_negative_columns(columns):
    valid = all(
        column in BUDGETED_COLUMNS and not isinstance(value, bool) and value >= 0
        for column, value in columns.items()
    )
    table = "".join(
        f"{column} = {str(value).lower()}\n" for column, value in columns.items()
    )
    with tempfile.TemporaryDirectory() as directory:
        budget_file = Path(directory) / "budget.toml"
        budget_file.write_text(f'["tests/module.py"]\n{table}')

        if valid:
            assert read_budget(budget_file)["tests/module.py"] == Budget(**columns)
        else:
            with pytest.raises(BudgetFileError, match="tests/module.py"):
                read_budget(budget_file)


@pytest.mark.no_fuzz
@pytest.mark.oracle_smoke
def test_a_cyclic_reexport_between_packages_terminates():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        outer = root / "SpectralSVR"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (outer / "__init__.py").write_text("from .inner import name as name\n")
        (inner / "__init__.py").write_text("from .. import name as name\n")

        exported = library_exports(root)

    assert "name" in exported[outer / "__init__.py"]


@pytest.mark.no_fuzz
@pytest.mark.oracle_independent
@pytest.mark.scenario("A docstring outside the shape is refused")
@pytest.mark.parametrize("name", [*SPEC_SECTIONS, *FREE_PROSE_SECTIONS])
def test_each_section_the_spec_names_is_admitted_and_no_other(name):
    body = (
        ">>> scale(1)" if name == "Examples" else f"value : float\n    {ONE_SENTENCE}"
    )
    docstring = f"{ONE_SENTENCE}\n\n{name}\n{'-' * len(name)}\n{body}"

    assert is_in_shape(docstring) == (name in SPEC_SECTIONS)
