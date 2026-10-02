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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

MACHINE_DIRECTIVE = re.compile(
    r"^#!|^#.*?coding[:=]\s*[-\w.]+"
    r"|#\s*(noqa|type:|pragma:|ruff:|mypy:|pyright:|pylint:|fmt:|isort:)",
    re.IGNORECASE,
)
LIBRARY = "SpectralSVR"
COUNTED_AREAS = ("SpectralSVR", "tests", "scripts")
DOCSTRING_SECTIONS = (
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
ABBREVIATIONS = (
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
INLINE_CODE = re.compile(r"`+[^`]*`+")
SENTENCE_END = re.compile(r"[.?!](?=\s|$)")
WORD = re.compile(r"\w")
DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
COMPOUND_STATEMENTS = (ast.If, ast.Try, ast.TryStar, ast.With, ast.AsyncWith)
BUDGETED_COLUMNS = ("comments", "internal_docstrings", "misshapen_docstrings")


@dataclass(frozen=True)
class Prose:
    """Counted prose in one file; ``public_docstrings`` is measured but never budgeted."""

    comments: int
    internal_docstrings: int
    public_docstrings: int
    misshapen_docstrings: int


@dataclass(frozen=True)
class Docstring:
    """One docstring as written, the lines its literal spans, and whether the interface carries it."""

    text: str
    first_line: int
    last_line: int
    public: bool

    @property
    def lines(self) -> int:
        return len(self.text.splitlines())


@dataclass(frozen=True)
class Budget:
    """The prose one file may carry; an unlisted file's budget is zero in every column."""

    comments: int = 0
    internal_docstrings: int = 0
    misshapen_docstrings: int = 0


class BudgetFileError(ValueError):
    """A budget file entry names an unknown column or a value that is not a non-negative integer."""


def comment_lines(source: str) -> int:
    return sum(
        1
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and not MACHINE_DIRECTIVE.search(token.string)
    )


def is_internal_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def unconditional_and_guarded(statements: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement at this level, including those under ``if``, ``try`` and ``with``."""
    for statement in statements:
        if not isinstance(statement, COMPOUND_STATEMENTS):
            yield statement
            continue
        for branch in ("body", "orelse", "finalbody"):
            yield from unconditional_and_guarded(getattr(statement, branch, []))
        for handler in getattr(statement, "handlers", []):
            yield from unconditional_and_guarded(handler.body)


def docstring_of(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    public: bool,
) -> list[Docstring]:
    text = ast.get_docstring(node, clean=False)
    if text is None:
        return []
    literal = node.body[0]
    return [
        Docstring(text, literal.lineno, literal.end_lineno or literal.lineno, public)
    ]


def definition_docstrings(
    statements: Iterable[ast.stmt],
    enclosing_public: bool,
    exported: frozenset[str] | None,
) -> Iterator[Docstring]:
    """Docstrings of definitions in ``statements``; ``exported`` filters only the outermost level."""
    for node in unconditional_and_guarded(statements):
        if not isinstance(node, DEFINITIONS):
            continue
        public = (
            enclosing_public
            and not is_internal_name(node.name)
            and (exported is None or node.name in exported)
        )
        yield from docstring_of(node, public)
        if isinstance(node, ast.ClassDef):
            yield from definition_docstrings(node.body, public, None)
            continue
        for nested in ast.walk(node):
            if nested is not node and isinstance(nested, DEFINITIONS):
                yield from docstring_of(nested, False)


def docstrings(source: str, exported: frozenset[str] | None) -> list[Docstring]:
    """Every docstring in ``source``; ``exported=None`` means the leading underscore alone decides."""
    tree = ast.parse(source)
    module_public = exported is None or bool(exported)
    return [
        *docstring_of(tree, module_public),
        *definition_docstrings(tree.body, True, exported),
    ]


def sentence_count(text: str) -> int:
    plain = INLINE_CODE.sub("code", text).replace("...", " ")
    for abbreviation in ABBREVIATIONS:
        plain = plain.replace(abbreviation, "abbreviation")
    plain = plain.strip()
    ends = list(SENTENCE_END.finditer(plain))
    trailing = plain[ends[-1].end() :] if ends else plain
    return len(ends) + bool(WORD.search(trailing))


def is_dash_underline(line: str) -> bool:
    return bool(line.strip()) and set(line.strip()) == {"-"}


def sections(body: list[str]) -> list[tuple[str | None, list[str]]]:
    """``body`` split at dash-underlined headers; text before the first header is under ``None``."""
    split: list[tuple[str | None, list[str]]] = [(None, [])]
    for line, following in itertools.pairwise([*body, ""]):
        if is_dash_underline(line):
            continue
        if is_dash_underline(following):
            split.append((line.strip(), []))
        else:
            split[-1][1].append(line)
    return split


def entry_descriptions(lines: list[str], see_also: bool) -> list[list[str]] | None:
    """Each entry's description lines, or ``None`` where indented text precedes every entry."""
    descriptions: list[list[str]] = []
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith(" "):
            _, _, remainder = line.partition(":")
            descriptions.append([remainder] if see_also else [])
        elif descriptions:
            descriptions[-1].append(line.strip())
        else:
            return None
    return descriptions


def is_doctest_only(lines: list[str]) -> bool:
    in_example = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_example = False
        elif stripped.startswith(">>>"):
            in_example = True
        elif not in_example:
            return False
    return True


def is_section_in_shape(name: str | None, lines: list[str]) -> bool:
    if name is None:
        return not any(line.strip() for line in lines)
    if name not in DOCSTRING_SECTIONS:
        return False
    if name == "Examples":
        return is_doctest_only(lines)
    descriptions = entry_descriptions(lines, see_also=name == "See Also")
    return descriptions is not None and all(
        sentence_count(" ".join(text)) <= 1 for text in descriptions
    )


def is_in_shape(docstring: str) -> bool:
    """Whether a docstring is one summary sentence plus closed-set sections of one-sentence entries."""
    lines = inspect.cleandoc(docstring).splitlines()
    summary_end = next(
        (n for n, line in enumerate(lines) if not line.strip()), len(lines)
    )
    summary = lines[:summary_end]
    if not summary or any(is_dash_underline(line) for line in summary):
        return False
    return sentence_count(" ".join(summary)) <= 1 and all(
        is_section_in_shape(name, section_lines)
        for name, section_lines in sections(lines[summary_end:])
    )


def module_file(package_directory: Path, level: int, module: str | None) -> Path:
    base = package_directory
    for _ in range(level - 1):
        base = base.parent
    target = base.joinpath(*module.split(".")) if module else base
    return target / "__init__.py" if target.is_dir() else target.with_suffix(".py")


def imported_module(root: Path, init: Path, node: ast.ImportFrom) -> Path | None:
    if node.level >= 1:
        return module_file(init.parent, node.level, node.module)
    if node.module == LIBRARY or (node.module or "").startswith(LIBRARY + "."):
        return module_file(root, 1, node.module)
    return None


def declared_all(statements: Iterable[ast.stmt]) -> set[str]:
    return {
        element.value
        for statement in statements
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in statement.targets
        )
        and isinstance(statement.value, (ast.List, ast.Tuple))
        for element in statement.value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }


def star_exports(module: Path) -> set[str]:
    statements = list(unconditional_and_guarded(ast.parse(module.read_text()).body))
    listed = declared_all(statements)
    if listed:
        return listed
    return {
        node.name
        for node in statements
        if isinstance(node, DEFINITIONS) and not is_internal_name(node.name)
    }


@dataclass(frozen=True)
class PackageInit:
    """What one package ``__init__`` binds: imported names with their sources, definitions and stars."""

    imports: dict[str, tuple[Path, str]]
    defined: set[str]
    stars: list[Path]
    listed: set[str]

    @classmethod
    def read(cls, root: Path, init: Path) -> "PackageInit":
        statements = list(unconditional_and_guarded(ast.parse(init.read_text()).body))
        imports: dict[str, tuple[Path, str]] = {}
        stars: list[Path] = []
        for node in statements:
            if not isinstance(node, ast.ImportFrom):
                continue
            source = imported_module(root, init, node)
            if source is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    stars.append(source)
                else:
                    imports[alias.asname or alias.name] = (source, alias.name)
        defined = {node.name for node in statements if isinstance(node, DEFINITIONS)}
        return cls(imports, defined, stars, declared_all(statements))


def library_exports(root: Path) -> dict[Path, frozenset[str]]:
    """Every library file mapped to the names its packages export from it."""
    inits = {
        init: PackageInit.read(root, init)
        for init in (root / LIBRARY).rglob("__init__.py")
    }

    def source_of(
        file: Path, name: str, seen: frozenset[tuple[Path, str]]
    ) -> tuple[Path, str]:
        package = inits.get(file)
        if (file, name) in seen or package is None or name not in package.imports:
            return file, name
        target, original = package.imports[name]
        return source_of(target, original, seen | {(file, name)})

    exported: dict[Path, set[str]] = {}
    for init, package in inits.items():
        bound = set(package.imports) | package.defined | package.listed
        for star in package.stars:
            names = star_exports(star) if star.exists() else set()
            exported.setdefault(star, set()).update(names)
            bound |= names
        exported.setdefault(init, set()).update(bound)
        for name in package.imports:
            file, original = source_of(init, name, frozenset())
            exported.setdefault(file, set()).add(original)
    return {file: frozenset(names) for file, names in exported.items()}


def ruff_docstring_report(root: Path) -> list[dict]:
    """Ruff's numpy-convention docstring findings over the counted areas under ``root``, as JSON records."""
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
    return json.loads(report.stdout)


def docstring_violation_rows(report: list[dict], root: Path) -> dict[str, set[int]]:
    """Rows of the docstring rule findings in ``report``, per file relative to ``root``."""
    flagged: dict[str, set[int]] = {}
    for violation in report:
        if not (violation.get("code") or "").startswith("D"):
            continue
        file = Path(violation["filename"]).relative_to(root.resolve()).as_posix()
        flagged.setdefault(file, set()).add(violation["location"]["row"])
    return flagged


def measure(root: Path) -> dict[str, Prose]:
    """Prose per file under every counted area, keyed by path relative to ``root``."""
    exports = library_exports(root)
    flagged = docstring_violation_rows(ruff_docstring_report(root), root)
    measured = {}
    for area in COUNTED_AREAS:
        for file in sorted((root / area).rglob("*.py")):
            source = file.read_text()
            relative = file.relative_to(root).as_posix()
            exported = exports.get(file, frozenset()) if area == LIBRARY else None
            found = docstrings(source, exported)
            rows = flagged.get(relative, set())
            measured[relative] = Prose(
                comments=comment_lines(source),
                internal_docstrings=sum(d.lines for d in found if not d.public),
                public_docstrings=sum(d.lines for d in found if d.public),
                misshapen_docstrings=sum(
                    not rows.isdisjoint(range(d.first_line, d.last_line + 1))
                    or not is_in_shape(d.text)
                    for d in found
                ),
            )
    return measured


def read_budget(path: Path) -> dict[str, Budget]:
    with path.open("rb") as handle:
        table = tomllib.load(handle)
    budget = {}
    for file, columns in table.items():
        unknown = sorted(set(columns) - set(BUDGETED_COLUMNS))
        invalid = sorted(
            column
            for column, value in columns.items()
            if not isinstance(value, int) or isinstance(value, bool) or value < 0
        )
        if unknown or invalid:
            raise BudgetFileError(
                f"{path}: {file} names unknown columns {unknown} and gives "
                f"{invalid} values that are not non-negative integers"
            )
        budget[file] = Budget(**columns)
    return budget


@dataclass(frozen=True)
class Breach:
    """One file's count in one column set against its allowance."""

    file: str
    column: str
    count: int
    allowance: int


def over_budget(measured: dict[str, Prose], budget: dict[str, Budget]) -> list[Breach]:
    return [
        Breach(
            file,
            column,
            getattr(prose, column),
            getattr(budget.get(file, Budget()), column),
        )
        for file, prose in sorted(measured.items())
        for column in BUDGETED_COLUMNS
        if getattr(prose, column) > getattr(budget.get(file, Budget()), column)
    ]


def slack(measured: dict[str, Prose], budget: dict[str, Budget]) -> list[Breach]:
    absent = Prose(
        comments=0, internal_docstrings=0, public_docstrings=0, misshapen_docstrings=0
    )
    return [
        Breach(
            file,
            column,
            getattr(measured.get(file, absent), column),
            getattr(allowance, column),
        )
        for file, allowance in sorted(budget.items())
        for column in BUDGETED_COLUMNS
        if getattr(allowance, column) > getattr(measured.get(file, absent), column)
    ]
