import importlib.util
import pathlib
import sys
import tempfile
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPOSITORY / "scripts" / "equivalent_mutants.py"
MODULE_NAME = "parity.py"
MODULE_SOURCE = "def is_even(modes):\n    return modes % 2 == 0\n"
EDITED_SOURCE = "def is_even(modes):\n    return modes % 2 != 1\n"
BLANK_REASONS = st.sampled_from(("", " ", "\t", "\n  "))
EXAMPLES = settings(max_examples=25, deadline=None)


def load_script():
    spec = importlib.util.spec_from_file_location("_equivalent_mutants", SCRIPT_PATH)
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    return script


script = load_script()


def written_module(directory: pathlib.Path, source: str) -> pathlib.Path:
    module = directory / MODULE_NAME
    module.write_text(source)
    return module


def entries_for_every_mutant(directory: pathlib.Path) -> list:
    module = written_module(directory, MODULE_SOURCE)
    source = module.read_text()
    return [
        script.Equivalent(
            module=MODULE_NAME,
            definition=mutation.definition_name or "",
            line=script.stripped_line(source, mutation.start_pos[0]),
            mutant=script.mutant_of(mutation, source),
            reason="stated",
        )
        for mutation in script.mutations_in(module)
    ]


@pytest.mark.oracle_invariant
@pytest.mark.no_fuzz
def test_every_declared_equivalent_matches_one_mutant_of_the_source():
    assert script.refusals(script.declared()) == []


@pytest.mark.oracle_invariant
@pytest.mark.scenario("An entry without a reason is refused")
@EXAMPLES
@given(choice=st.integers(min_value=0), reason=BLANK_REASONS)
def test_an_entry_without_a_reason_is_refused(choice, reason):
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        entries = entries_for_every_mutant(directory)
        entry = replace(entries[choice % len(entries)], reason=reason)
        assert script.refusals([entry], root=directory) == [
            f"{entry}: states no reason"
        ]


@pytest.mark.oracle_invariant
@pytest.mark.scenario("The declared line was edited")
@EXAMPLES
@given(choice=st.integers(min_value=0))
def test_an_entry_whose_line_was_edited_is_refused(choice):
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        entries = entries_for_every_mutant(directory)
        entry = entries[choice % len(entries)]
        written_module(directory, EDITED_SOURCE)
        assert script.refusals([entry], root=directory) == [
            f"{entry}: no mutated line of {MODULE_NAME} reads as its line"
        ]


@pytest.mark.oracle_invariant
@pytest.mark.scenario("The declared mutant no longer arises")
@EXAMPLES
@given(choice=st.integers(min_value=0), suffix=st.text(min_size=1, max_size=8))
def test_an_entry_whose_mutant_no_longer_arises_is_refused(choice, suffix):
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        entries = entries_for_every_mutant(directory)
        entry = entries[choice % len(entries)]
        mutants = {
            candidate.mutant for candidate in entries if candidate.line == entry.line
        }
        unmatched = replace(entry, mutant=f"{entry.mutant}  # {suffix}")
        assert unmatched.mutant not in mutants
        assert script.refusals([unmatched], root=directory) == [
            f"{unmatched}: 0 mutants of its line read as declared"
        ]


@pytest.mark.oracle_invariant
@pytest.mark.scenario("A declared mutant is skipped")
@pytest.mark.scenario("Other mutants on the same line still run")
@EXAMPLES
@given(choice=st.integers(min_value=0))
def test_only_the_declared_mutant_is_skipped(choice):
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        entries = entries_for_every_mutant(directory)
        entry = entries[choice % len(entries)]
        session_path = directory / "session.sqlite"
        with script.use_db(session_path) as session:
            script.init([directory / MODULE_NAME], session, {})
            every_job = {item.job_id for item in session.work_items}

        assert script.skip_declared(session_path, [entry], root=directory) == 1

        with script.use_db(session_path, script.WorkDB.Mode.open) as session:
            still_pending = {item.job_id for item in session.pending_work_items}
        others_on_its_line = [
            candidate
            for candidate in entries
            if candidate.line == entry.line and candidate != entry
        ]
        assert others_on_its_line
        assert len(every_job - still_pending) == 1
        assert len(still_pending) == len(entries) - 1
