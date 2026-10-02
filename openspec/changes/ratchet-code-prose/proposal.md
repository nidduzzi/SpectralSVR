# Ratchet the prose that lives in code

## Why

`AGENTS.md` states the rule: "Express intent through names and structure.
Contracts belong in docstrings; explanatory prose comments in source do not."
Nothing enforces it, so the rule holds only where a reviewer happens to look.

Two review rounds on `time-dependent-forcing-in-numerical-burgers` found the
same failure twice, in the same file, after it had already been corrected once:

- A comment block recording a convergence table was rewritten twice. The first
  revision had the wrong orders, because it read `log2` of an error ratio on a
  sweep whose steps did not halve. Nothing checked it, because a comment is not
  executable. The numbers it carried now live in the change's `tasks.md`, where
  they are the record and where a wrong figure is a wrong claim rather than a
  stale note.
- Three docstrings re-derived arithmetic that the change's `design.md` already
  owned — `ns >= 6 K` from the Orszag mask, `2 K <= modes / 3` for the widest
  manufactured band, and why an over-wide band is refused at generation. Each
  was cut. Each had passed review when written.

Neither was caught by `just check`. Both were caught by a reviewer reading
carefully, which does not scale and did not catch them the first time.

The current surface, measured with `tokenize` and `ast`, pragmas excluded:

| area | files | comment lines | docstring (public) | docstring (private) |
| --- | ---: | ---: | ---: | ---: |
| `SpectralSVR` | 28 | 229 | 1229 | 123 |
| `tests` | 24 | 467 | 198 | 54 |
| `scripts` | 1 | 12 | 60 | 0 |

The worst files are `tests/problems/test_problems.py` (82 comment lines),
`tests/basis/test_fourier_basis.py` (77), `tests/basis/test_coeff_anchors.py`
(58) and `SpectralSVR/basis/FourierBasis.py` (49).

## What Changes

- A guard test holding a **per-file budget** for comment lines and for private
  docstring lines, asserted `<=` and shrinking only. Modelled on
  `tests/test_scenario_coverage.py`'s `EXEMPT`: debt with a number against it,
  never furniture.
- A **fixed docstring shape** that leaves a derivation no room: one summary
  sentence, then only numpydoc sections from a closed set, each entry at most one
  sentence. A derivation is a chain of sentences; the shape has nowhere to put
  one, and a parser checks it without judging content. Existing docstrings
  outside the shape are counted per file and the count only shrinks.
- The budget is seeded from today's measurement, so the guard is green on
  arrival and every later commit either holds or shrinks it.

Explicitly **not** ratcheted: public docstring lines. Those are the library's
interface for the readers of the paper it backs, and 1229 lines across 28 files
is not the problem this change found. A length budget on them would push toward
dense and unreadable rather than toward absent, optimising the wrong thing.
Public docstrings are held to the shape, though, since that is where one of the
two observed derivations sat.

## Capabilities

### New Capabilities
- `code-prose-budget`: what counts as prose in code, how the budget is measured
  and enforced, which surfaces are exempt and why, and how the budget moves.

## Impact

- New guard in `tests/`, alongside `test_oracle_coverage.py` and
  `test_scenario_coverage.py`, which it deliberately resembles.
- A budget file recording per-file counts. Every entry is debt.
- No library source changes. No dependency: `tokenize` gives comments and `ast`
  gives docstrings, both standard library. `ast-grep` was considered and is the
  wrong tool here — `ast` discards comments entirely, so a syntax-tree matcher
  cannot see them.
- No reported number moves; nothing executable changes.
