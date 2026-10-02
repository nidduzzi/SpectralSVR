## Context

See proposal.md - Why. The repository already runs two guards of this shape,
and this one is deliberately the third:

- `tests/test_oracle_coverage.py` — every module carries an anchor, with a short
  `EXEMPT` set.
- `tests/test_scenario_coverage.py` — every stated scenario has a test or a
  reason, with `EXEMPT` as debt keyed per scenario.

Both work because what they check is objective: a marker is present or it is
not. The hard part of this change is that "prose" is not objective, and the
design's job is to find the parts that are.

Measured surface, `tokenize` for comments and `ast` for docstrings, pragmas
excluded:

| area | files | comment | doc (public) | doc (private) |
| --- | ---: | ---: | ---: | ---: |
| `SpectralSVR` | 28 | 229 | 1229 | 123 |
| `tests` | 24 | 467 | 198 | 54 |
| `scripts` | 1 | 12 | 60 | 0 |

*Corrected during apply:* the library and script rows could not be reproduced on any tree
this census could have come from, and its method was never recorded; only the `tests` row
reproduces exactly. The counts at the branch head are recorded in tasks 1.1 and 1.2.

## Goals / Non-Goals

**Goals:**

- Make the `AGENTS.md` prose rule checkable, so it does not depend on a reviewer
  reading carefully.
- Bound internal narration, and let the bound only fall.
- Refuse the specific failure observed twice: a derivation copied into a
  docstring from a change's design.

**Non-Goals:**

- Judging whether prose is *good*. Any proxy for quality is gameable toward
  something worse, and this design does not attempt one.
- Bounding public docstrings. They are the interface for readers of the paper
  this library backs.
- Removing the existing prose. The budget is seeded from what is there; shrinking
  it is later work, done by the changes that touch those files.

## Decisions

### Count with `tokenize` and `ast`, not with `ast-grep`

`ast-grep` matches syntax trees. Python's parser discards comments before a tree
exists — they are not nodes — so no tree matcher can see them. `tokenize`
yields `COMMENT` tokens directly, and `ast.get_docstring` yields docstrings.
Both are standard library. *Corrected during apply:* the docstring shape added
ruff, declared in the test extra, and its numpy convention in `pyproject.toml`.

*Alternative rejected:* a regex over source lines. It cannot tell a `#` inside a
string literal from a comment, and this repository has string literals
containing `#` in error messages and docstrings.

### Per-file budgets, and an unlisted file is budgeted at zero

A repository total lets one file's cleanup pay for another file's narration,
which is exactly the trade this change exists to refuse. Per-file makes each
addition visible where it happens.

Defaulting an unlisted file to zero rather than unbounded is what makes the
guard hold for code written after it: a new file cannot quietly acquire an
allowance, and adding a file to the budget is a deliberate act with a number
attached.

### The budget reports slack rather than silently tolerating it

`<=` alone lets a file sit under its budget forever, leaving room that a later
commit spends without the guard noticing. The second assertion — that no budget
exceeds its file's actual count — is what turns the list into a ratchet rather
than a ceiling. It is the same reasoning as `test_no_exemption_is_stale`, which
refuses an `EXEMPT` entry that has quietly gained a test.

Cost: every prose deletion forces a budget edit in the same commit. That is the
point; the number moving down in the diff is the record that it moved.

### Public versus internal is read from structure, never from a list

A leading underscore, or absence from the package's `__init__` re-exports. Both
are readable with `ast` and neither is a judgement someone maintains. A
per-name allowance list would reintroduce the argument this change removes: it
would become the place to put "this one is worth it".

*Alternative rejected:* a decorator or marker comment declaring a docstring
public. It puts the decision back in the author's hands at the point where they
are motivated to keep the prose.

### Derivations are made structurally impossible by the docstring's shape

Decided by the user during apply. A docstring is one summary sentence, then
optionally numpydoc sections from a closed set -- Parameters, Returns, Yields,
Raises, Warns, Attributes, See Also, Examples -- each entry's description at most
one sentence. Notes and free paragraphs are not in the set. A derivation is a
chain of sentences; the shape has nowhere to put one. Conditions a caller must
meet go in a Parameters or Raises entry, as the condition alone.

The check parses: it splits the docstring into a summary paragraph and sections
by their numpydoc headers and underlines, splits each section into entries, and
counts sentence terminators outside inline code (``...``) and outside a closed
list of abbreviations (e.g., i.e., etc.). Examples bodies are code and are not
counted. A docstring outside the shape counts against a third budget column,
`misshapen_docstrings`, seeded from the head and shrink-only like the others.
Public docstrings are held to the shape: one of the two observed derivations sat
in a public one, and the exemption existed to spare the interface a length
limit, not to let it carry reasoning.

Packing a chain into one long sentence stays possible. It is visible, bounded to
one sentence per entry, and review sees it; the shape removes the room that let
two multi-sentence derivations pass review, which is the failure this change was
written for.

*Alternative rejected: a content heuristic.* Measured during apply against every
docstring in the repository and the two known-bad docstrings recoverable from
`4e2b786`'s parent (`_solve_spectral` and `widest_manufactured_band` in
`SpectralSVR/problems/burgers.py`). The first candidate, two or more inline
expressions with an operator plus a reasoning word, caught both and fired on 18
of 243 current docstrings, most of them contracts. The best tuned rule -- three
sentences pairing a reasoning word with an operator expression, or two with a
strong connective such as "therefore" -- caught both and fired on none, but it
was fitted to two positives, and three current contracts (`as_semilinear_solver`,
`burgers_periodic_manufactured`, `_solve_spectral`) sat one sentence below its
threshold. A rule that recognises only the shape it was tuned on is not a
guarantee, and the user rejected it as a heuristic.

## Risks / Trade-offs

- **The shape rejects docstrings that are fine as prose.** Many public
  docstrings run to paragraphs today. → They are recorded as debt in the third
  column rather than rewritten here; each shrinks when its file is next touched.

- **Sentence counting has edge cases.** Abbreviations and decimals contain
  periods. → Terminators inside inline code are ignored, the abbreviation list is
  closed and stated, and the parser is fuzzed over constructed docstrings with
  those cases pinned.

- **A budget edit in every prose-touching commit is friction.** → Accepted, and
  it is the mechanism rather than a side effect. The alternative is a ceiling
  that never falls.

- **`tests/` carries twice the library's comment lines.** Some record
  measurements that have nowhere better to live once a change is archived. →
  Open question below; this change must not force those into deletion before it
  is settled where they go.

- **The guard measures prose, which is not the same as intent.** A file can pass
  with prose that says nothing and fail with prose that helps. → Stated plainly:
  this bounds quantity, and review still owns quality. It closes the failure that
  review missed twice, not the whole rule.

## Migration Plan

1. Write the counter and verify it against constructed modules of known
   counts and against the `tests` census, the one row that reproduces.
   *Corrected during apply:* first written to treat any disagreement with the
   whole census as a counter bug, which the unreproducible rows made untenable.
2. Seed the budget from the measurement so the guard is green on arrival.
3. Hold every docstring to the numpy shape with this change's constraint.
   *Corrected during apply:* first written as tuning a derivation heuristic,
   which the user rejected.
4. Wire into `just check`.

Rollback is deleting the guard and its budget; nothing else depends on it.

## Decided during apply

**A recorded measurement beside an assertion counts as prose.** Decided by the
user at task 0.1. One rule holds everywhere: a measured figure lives in the
change's `tasks.md` or `measurements/`, and where it backs a bound it is a
named constant the assertion uses, such as `MEASURED_WORST = 2.06e-4`, so the
number stays beside the assertion as code rather than as a comment and cannot
drift from it silently. This is what `time-dependent-forcing-in-numerical-burgers`
ended up doing. It barely bites on the hardest file:
`tests/problems/test_problems.py` carries 82 comment lines, of which 2 carry a
figure or a measurement word; the rest are section banners and narration.

**Section banners are counted.** Decided by the user at the same point. A
`# ==== #` block and its title are comments, and the budget counts them like any
other; structure belongs to test classes, state machines or separate modules.
The seeded budget absorbs today's banners, and they shrink as files are touched.

**The shape follows numpy style, the convention already in use.** Decided by the
user when the shape was implemented. Of the 43 docstrings with sections, 12
written by recent changes use numpy style, 31 in the older model and basis code
use a nonstandard `Arguments:` with `name {type} --` form, and 2 use Google style;
numpy is also the scientific-Python norm. The 31 count as misshapen debt and
convert when their files are touched.

**Standard checks come from ruff, and the guard owns the ratchet.** Decided by
the user. `pyproject.toml` states `convention = "numpy"` for ruff's pydocstyle
rules, but `D` stays out of the default selection: ruff cannot report a
per-file ignore that no longer matches, so grandfathering there would never
shrink. The guard runs `ruff check --select D --ignore D1` and joins each
violation to the docstring whose literal starts on its line; a docstring is
misshapen when ruff flags it or when it breaks this change's constraint. The
missing-docstring rules stay off, since this change wants less prose. Ruff is
declared in the test extra, where the guard runs.

**The closed set admits numpydoc's entry-list sections; Examples holds only doctests.**
Decided by the user after the review. The closed set is Parameters, Other
Parameters, Receives, Returns, Yields, Raises, Warns, Attributes, Methods, See
Also and Examples: every numpydoc section made of one-sentence entries. Warnings,
References and Notes stay out, because each is free prose. An Examples section
may hold only doctest lines, `>>>` and `...`, and the output that follows them;
prose there would carry a derivation past the one-sentence rule.

**The review's findings are all in scope.** Decided by the user. They are listed
as tasks 6.1 to 6.14; each closes a misclassification the review proved with a
probe, or anchors a rule that only a consistency check held.

