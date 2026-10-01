## 0. Settle the open question first

- [x] 0.1 Decide whether a recorded measurement beside an assertion counts as prose, and record the decision in design.md before any budget is seeded for `tests/`. `AGENTS.md` asks that a measured tolerance be recorded where the bound is set; `time-dependent-forcing-in-numerical-burgers` resolved it by moving every figure into the change's `tasks.md` and deleting the comments, which worked but cost the numbers their proximity to the assertion they justify. Either that becomes the rule, or "a measurement record" becomes a permitted kind with a definition narrow enough not to be a loophole. Verify by applying the chosen rule to `tests/problems/test_problems.py`, whose 82 comment lines are the hardest case in the repository
  - Decided by the user: a recorded measurement counts as prose, and lives in the change's artifacts or as a named constant the assertion uses; section banners are counted too. Recorded in design.md under "Decided during apply". Applied to `tests/problems/test_problems.py`: all 82 comment lines count, of which 2 carry a figure or a measurement word; the rest are banners and narration.

## 1. Counting

- [ ] 1.1 Count comment lines per file with `tokenize`, excluding machine-directed comments (`noqa`, `type:`, `pragma:`, `ruff:`, `mypy:`, `pyright:`, `fmt:`, `isort:`, shebang); verify against constructed modules whose counts are known by construction, fuzzed per 4.2, and against the census for `tests` (467), which reproduces exactly. *Amended during apply, at the user's direction:* first written to reproduce the proposal's library census exactly, which cannot be done on any tree it could have come from -- on the original proposal tree, `c724686`, library comments are 219 prose, 9 `pragma:`, 3 `pyright:`, 3 `noqa` and 2 `type:`, and no subset sums to the stated 229 -- so the census method was never recorded and it is not an oracle. Record the head's counts
- [ ] 1.2 Count docstring lines per file with `ast`, as the physical lines the docstring spans. In `SpectralSVR` a name is public when it carries no leading underscore and its package exports it, by re-export from an `__init__` or by definition in one; methods follow their class, nested definitions are internal, and a module docstring is public when the module exports a name. In `tests` and `scripts`, which export nothing, the leading underscore alone decides. Verify against constructed modules and export maps, and against the `tests` census (198 public, 54 internal). *Amended during apply:* the library split of 1229 against 123 cannot be reproduced by any reading either, and the export rule, decided by the user to follow the spec, moves unexported library names into the budget
- [ ] 1.3 Verify the counter is blind to `#` inside string literals and to `"""` inside comments, by constructing both cases and asserting the count is unchanged

## 2. The budget

- [ ] 2.1 Record the per-file budget seeded from 1.1 and 1.2, with internal docstring lines and comment lines as separate columns so a file's two debts are visible apart; verify the guard is green on arrival with no file over budget
- [ ] 2.2 Fail a file that exceeds its budget, naming the file, the budget and the count; verify by adding a comment line to a budgeted file and confirming the guard names it
- [ ] 2.3 Fail a file carrying counted prose with no budget entry, so an unlisted file is budgeted at zero rather than unbounded; verify with a new file carrying one comment
- [ ] 2.4 Fail a budget that exceeds its file's actual count, so slack is reclaimed rather than banked -- the same shape as `test_no_exemption_is_stale`; verify by deleting a comment without lowering the budget and confirming the guard forces the edit
- [ ] 2.5 Verify the budget is per file rather than pooled: add a comment to one file, remove one from another, confirm the guard still fails

## 3. Derivations in docstrings

- [ ] 3.1 Measure before asserting: run a candidate check over every docstring in the repository and record two sets in design.md -- what it catches and what it fires on wrongly. The archived `manufactured-band-derivation` and `time-dependent-forcing-in-numerical-burgers` supply known-bad examples, since three docstrings cut during their reviews are recoverable from git history
- [ ] 3.2 Tune until the false-positive set is empty and the known-bad set is caught, and record the final rule with both sets; a check that fires on legitimate contract text gets suppressed rather than obeyed, so an untuned check is worse than none
- [ ] 3.3 Fail a docstring carrying a derivation, naming it; verify against a known-bad example from 3.1
- [ ] 3.4 Accept a docstring stating a bound or condition without reconstructing the argument for it; verify against the contract-only docstrings the same files now carry

## 4. Scenario coverage

- [ ] 4.1 Mark each of the capability's scenarios with `@pytest.mark.scenario` as its task completes and remove its `IN_FLIGHT` entry in the same commit. The nine scenarios enter `EXEMPT` when this change is proposed, because it is parked at the stack tip with its artifacts written and no implementation, and a stated scenario with neither a test nor an entry leaves the suite red. Verify the stale-exemption test forces each removal rather than tolerating it
- [ ] 4.2 Fuzz the counting properties over generated sources rather than pinning one file: comment and docstring counts over constructed modules, and the public/internal split over generated names. Verify each property fails under a mutation of the behaviour it describes
- [ ] 4.3 Verify no scenario of this capability sits in `EXEMPT` at the end, so the nine entries added at proposal time are all discharged rather than archived as debt

## 5. Verification

- [ ] 5.1 Wire the guard into `just check`; verify clean
- [ ] 5.2 Verify the guard adds no dependency: `tokenize` and `ast` are standard library, and `pyproject.toml` and `uv.lock` are untouched
- [ ] 5.3 Verify the guard itself obeys the rule it enforces -- its own comment and internal-docstring lines are budgeted like any other file, with no exemption for being the guard
- [ ] 5.4 Verify no reported number moves and nothing executable changes, by diff argument: this change adds a test and a data file and touches no library source
