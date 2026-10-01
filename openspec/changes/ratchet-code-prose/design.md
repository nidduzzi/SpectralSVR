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
Both are standard library, so the guard adds no dependency and does not touch
`pyproject.toml`.

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

### Derivations are refused on content, and the check is narrow on purpose

The observed failures all had the same shape: arithmetic reconstructing why a
bound holds, inside a docstring, duplicating a change's design. That is
detectable without judging style — a docstring stating `ns >= 6 K` is a
contract, a docstring walking from the mask's `modes / 3` through the product's
`2 K` to reach it is a derivation.

A narrow check that catches this category and nothing else is worth more than a
broad one that fires on legitimate contract text and gets suppressed. Where the
boundary sits is the one thing this design cannot settle in advance, so tasks
below require it be measured against the docstrings already in the repository
and tuned until the false-positive set is empty and the known-bad set is caught.

## Risks / Trade-offs

- **The derivation check could fire on a legitimate contract.** → Tuned against
  the whole repository before it is asserted, with both sets recorded. A check
  nobody trusts is a check someone suppresses.

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

1. Write the counter and verify it against the census numbers in proposal.md;
   any disagreement is a counter bug, not a census correction.
2. Seed the budget from the measurement so the guard is green on arrival.
3. Tune the derivation check against every docstring in the repository.
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
