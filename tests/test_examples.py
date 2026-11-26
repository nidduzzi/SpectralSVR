import os
import pathlib
import subprocess

import nbformat
import pytest
from hypothesis import given
from hypothesis import strategies as st
from nbclient import NotebookClient

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES_DIRECTORY = REPOSITORY / "examples"
EXAMPLES = sorted(EXAMPLES_DIRECTORY.glob("*.ipynb"))
CELL_SECONDS = 120
NO_ACCELERATOR = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")


def stale_cells(notebook: nbformat.NotebookNode) -> list[int]:
    return [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and (cell.outputs or cell.execution_count is not None)
    ]


def misplaced(tracked_paths: list[str]) -> list[str]:
    return [
        path
        for path in tracked_paths
        if path.endswith(".ipynb")
        and pathlib.PurePosixPath(path).parts[0] != "examples"
    ]


def tracked_paths() -> list[str]:
    return subprocess.run(
        ["git", "ls-files"], cwd=REPOSITORY, capture_output=True, text=True, check=True
    ).stdout.splitlines()


@pytest.mark.oracle_smoke
@pytest.mark.no_fuzz
@pytest.mark.scenario("An example breaks against the library")
@pytest.mark.scenario("An example needs an accelerator")
@pytest.mark.parametrize(
    "example", EXAMPLES, ids=[example.stem for example in EXAMPLES]
)
def test_every_example_runs_on_cpu(example):
    notebook = nbformat.read(example, as_version=4)
    NotebookClient(
        notebook,
        timeout=CELL_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES_DIRECTORY)}},
    ).execute(env=NO_ACCELERATOR)


@pytest.mark.oracle_invariant
@pytest.mark.no_fuzz
@pytest.mark.scenario("A notebook with outputs is committed")
@pytest.mark.scenario("A notebook is added elsewhere")
def test_examples_are_the_only_notebooks_and_carry_no_outputs():
    assert misplaced(tracked_paths()) == []
    assert {
        example.name: stale_cells(nbformat.read(example, as_version=4))
        for example in EXAMPLES
    } == {example.name: [] for example in EXAMPLES}


CELLS = st.lists(
    st.tuples(
        st.sampled_from(("code", "markdown")),
        st.booleans(),
        st.one_of(st.none(), st.integers(min_value=1, max_value=99)),
    ),
    max_size=12,
)


@pytest.mark.oracle_analytic
@given(cells=CELLS)
def test_a_cell_with_an_output_or_an_execution_count_is_stale(cells):
    notebook = nbformat.v4.new_notebook()
    for cell_type, has_output, execution_count in cells:
        if cell_type == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell("text"))
        else:
            cell = nbformat.v4.new_code_cell("1", execution_count=execution_count)
            if has_output:
                cell.outputs = [nbformat.v4.new_output("stream", text="1")]
            notebook.cells.append(cell)
    assert stale_cells(notebook) == [
        index
        for index, (cell_type, has_output, execution_count) in enumerate(cells)
        if cell_type == "code" and (has_output or execution_count is not None)
    ]


SEGMENTS = st.text(alphabet="abcdefghij_-", min_size=1, max_size=6)


@pytest.mark.oracle_analytic
@given(
    directories=st.lists(st.one_of(st.just("examples"), SEGMENTS), max_size=3),
    stem=SEGMENTS,
    suffix=st.sampled_from((".ipynb", ".py", ".md")),
)
def test_a_notebook_outside_examples_is_misplaced(directories, stem, suffix):
    path = "/".join([*directories, stem + suffix])
    expected = [path] if suffix == ".ipynb" and directories[:1] != ["examples"] else []
    assert misplaced([path]) == expected
