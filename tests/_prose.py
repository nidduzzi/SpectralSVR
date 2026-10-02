"""How much prose a source file carries: comment lines, and docstring lines by visibility."""

import ast
import inspect
import io
import itertools
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path

MACHINE_DIRECTIVE = re.compile(
    r"^#!|#\s*(noqa|type:|pragma:|ruff:|mypy:|pyright:|fmt:|isort:)"
)
LIBRARY = "SpectralSVR"
COUNTED_AREAS = ("SpectralSVR", "tests", "scripts")


DOCSTRING_SECTIONS = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Attributes",
    "See Also",
    "Examples",
)
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "cf.", "vs.")
INLINE_CODE = re.compile(r"`+[^`]*`+")
SENTENCE_END = re.compile(r"[.?!](?=\s|$)")
DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Prose:
    """Counted prose in one file; ``public_docstrings`` is measured but never budgeted."""

    comments: int
    internal_docstrings: int
    public_docstrings: int
    misshapen_docstrings: int


@dataclass(frozen=True)
class Docstring:
    """One docstring as written, the line its literal starts on, and whether the interface carries it."""

    text: str
    line: int
    public: bool

    @property
    def lines(self) -> int:
        return len(self.text.splitlines())


def comment_lines(source: str) -> int:
    return sum(
        1
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and not MACHINE_DIRECTIVE.search(token.string)
    )


def is_internal_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def docstrings(source: str, exported: frozenset[str] | None) -> list[Docstring]:
    """Every docstring in ``source``; ``exported=None`` means the leading underscore alone decides."""
    tree = ast.parse(source)
    found = []
    module_text = ast.get_docstring(tree, clean=False)
    if module_text is not None:
        found.append(
            Docstring(
                module_text, tree.body[0].lineno, exported is None or bool(exported)
            )
        )
    for node in tree.body:
        if not isinstance(node, DEFINITIONS):
            continue
        node_public = not is_internal_name(node.name) and (
            exported is None or node.name in exported
        )
        for inner in ast.walk(node):
            if not isinstance(inner, DEFINITIONS):
                continue
            text = ast.get_docstring(inner, clean=False)
            if text is None:
                continue
            member_of_public_class = (
                isinstance(node, ast.ClassDef)
                and inner in node.body
                and not is_internal_name(inner.name)
            )
            public = node_public and (inner is node or member_of_public_class)
            found.append(Docstring(text, inner.body[0].lineno, public))
    return found


def sentence_count(text: str) -> int:
    plain = INLINE_CODE.sub("code", text)
    for abbreviation in ABBREVIATIONS:
        plain = plain.replace(abbreviation, "abbreviation")
    return len(SENTENCE_END.findall(plain.strip()))


def is_dash_underline(line: str) -> bool:
    return bool(line.strip()) and set(line.strip()) == {"-"}


def section_entry_descriptions(body: list[str]) -> list[list[str]] | None:
    """Each section entry's description lines, or ``None`` where ``body`` is not only closed-set sections."""
    section = None
    descriptions: list[list[str]] = []
    for line, following in itertools.pairwise([*body, ""]):
        if not line.strip() or is_dash_underline(line):
            continue
        if is_dash_underline(following):
            if line.strip() not in DOCSTRING_SECTIONS:
                return None
            section = line.strip()
        elif section is None:
            return None
        elif section == "Examples":
            continue
        elif not line.startswith(" "):
            _, _, remainder = line.partition(":")
            descriptions.append([remainder] if section == "See Also" else [])
        elif descriptions:
            descriptions[-1].append(line.strip())
        else:
            return None
    return descriptions


def is_in_shape(docstring: str) -> bool:
    """Whether a docstring is one summary sentence plus sections whose entries are one sentence each."""
    lines = inspect.cleandoc(docstring).splitlines()
    summary_end = next(
        (n for n, line in enumerate(lines) if not line.strip()), len(lines)
    )
    summary = lines[:summary_end]
    if not summary or any(is_dash_underline(line) for line in summary):
        return False
    descriptions = section_entry_descriptions(lines[summary_end:])
    return (
        sentence_count(" ".join(summary)) <= 1
        and descriptions is not None
        and all(sentence_count(" ".join(text)) <= 1 for text in descriptions)
    )


def module_file(package_directory: Path, level: int, module: str | None) -> Path:
    base = package_directory
    for _ in range(level - 1):
        base = base.parent
    target = base.joinpath(*module.split(".")) if module else base
    return target / "__init__.py" if target.is_dir() else target.with_suffix(".py")


def library_exports(root: Path) -> dict[Path, frozenset[str]]:
    """Every library file mapped to the names its packages export from it."""
    imports: dict[Path, dict[str, tuple[Path, str]]] = {}
    defined: dict[Path, set[str]] = {}
    for init in (root / LIBRARY).rglob("__init__.py"):
        tree = ast.parse(init.read_text())
        imports[init] = {
            (alias.asname or alias.name): (
                module_file(init.parent, node.level, node.module),
                alias.name,
            )
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.level >= 1
            for alias in node.names
        }
        defined[init] = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def source_of(
        file: Path, name: str, seen: frozenset[tuple[Path, str]]
    ) -> tuple[Path, str]:
        if (file, name) in seen or name not in imports.get(file, {}):
            return file, name
        target, original = imports[file][name]
        return source_of(target, original, seen | {(file, name)})

    exported: dict[Path, set[str]] = {}
    for init, names in imports.items():
        for name in names:
            file, original = source_of(init, name, frozenset())
            exported.setdefault(file, set()).add(original)
    for init in imports:
        exported.setdefault(init, set()).update(imports[init], defined[init])
    return {file: frozenset(names) for file, names in exported.items()}


def numpy_convention_violations(root: Path) -> dict[str, set[int]]:
    """Lines where ruff's numpy-convention docstring rules flag a docstring, per file relative to ``root``."""
    report = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *COUNTED_AREAS,
            "--select",
            "D",
            "--ignore",
            "D1",
            "--output-format",
            "json",
            "--exit-zero",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    flagged: dict[str, set[int]] = {}
    for violation in json.loads(report.stdout):
        file = Path(violation["filename"]).relative_to(root.resolve()).as_posix()
        flagged.setdefault(file, set()).add(violation["location"]["row"])
    return flagged


def measure(root: Path) -> dict[str, Prose]:
    """Prose per file under every counted area, keyed by path relative to ``root``."""
    exports = library_exports(root)
    flagged = numpy_convention_violations(root)
    measured = {}
    for area in COUNTED_AREAS:
        for file in sorted((root / area).rglob("*.py")):
            source = file.read_text()
            relative = file.relative_to(root).as_posix()
            exported = exports.get(file, frozenset()) if area == LIBRARY else None
            found = docstrings(source, exported)
            measured[relative] = Prose(
                comments=comment_lines(source),
                internal_docstrings=sum(d.lines for d in found if not d.public),
                public_docstrings=sum(d.lines for d in found if d.public),
                misshapen_docstrings=sum(
                    d.line in flagged.get(relative, set()) or not is_in_shape(d.text)
                    for d in found
                ),
            )
    return measured


@dataclass(frozen=True)
class Budget:
    """The prose one file may carry; an unlisted file's budget is zero in every column."""

    comments: int = 0
    internal_docstrings: int = 0
    misshapen_docstrings: int = 0


BUDGETED_COLUMNS = ("comments", "internal_docstrings", "misshapen_docstrings")


def read_budget(path: Path) -> dict[str, Budget]:
    with path.open("rb") as handle:
        return {
            file: Budget(**columns) for file, columns in tomllib.load(handle).items()
        }


def over_budget(measured: dict[str, Prose], budget: dict[str, Budget]) -> list[str]:
    return [
        f"{file}: {column} {getattr(prose, column)} over its budget of "
        f"{getattr(budget.get(file, Budget()), column)}"
        for file, prose in sorted(measured.items())
        for column in BUDGETED_COLUMNS
        if getattr(prose, column) > getattr(budget.get(file, Budget()), column)
    ]


def slack(measured: dict[str, Prose], budget: dict[str, Budget]) -> list[str]:
    absent = Prose(
        comments=0, internal_docstrings=0, public_docstrings=0, misshapen_docstrings=0
    )
    return [
        f"{file}: {column} budget {getattr(allowance, column)} is above the "
        f"{getattr(measured.get(file, absent), column)} the file carries; lower it"
        for file, allowance in sorted(budget.items())
        for column in BUDGETED_COLUMNS
        if getattr(allowance, column) > getattr(measured.get(file, absent), column)
    ]
