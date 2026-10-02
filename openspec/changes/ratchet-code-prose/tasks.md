## 0. Settle the open question first

- [x] 0.1 Decide whether a recorded measurement beside an assertion counts as prose, and record the decision in design.md before any budget is seeded for `tests/`. `AGENTS.md` asks that a measured tolerance be recorded where the bound is set; `time-dependent-forcing-in-numerical-burgers` resolved it by moving every figure into the change's `tasks.md` and deleting the comments, which worked but cost the numbers their proximity to the assertion they justify. Either that becomes the rule, or "a measurement record" becomes a permitted kind with a definition narrow enough not to be a loophole. Verify by applying the chosen rule to `tests/problems/test_problems.py`, whose 82 comment lines are the hardest case in the repository
  - Decided by the user: a recorded measurement counts as prose, and lives in the change's artifacts or as a named constant the assertion uses; section banners are counted too. Recorded in design.md under "Decided during apply". Applied to `tests/problems/test_problems.py`: all 82 comment lines count, of which 2 carry a figure or a measurement word; the rest are banners and narration.

## 1. Counting

- [x] 1.1 Count comment lines per file with `tokenize`, excluding machine-directed comments (`noqa`, `type:`, `pragma:`, `ruff:`, `mypy:`, `pyright:`, `fmt:`, `isort:`, shebang); verify against constructed modules whose counts are known by construction, fuzzed per 4.2, and against the census for `tests` (467), which reproduces exactly. *Amended during apply, at the user's direction:* first written to reproduce the proposal's library census exactly, which cannot be done on any tree it could have come from -- on the original proposal tree, `c724686`, library comments are 219 prose, 9 `pragma:`, 3 `pyright:`, 3 `noqa` and 2 `type:`, and no subset sums to the stated 229 -- so the census method was never recorded and it is not an oracle. Record the head's counts
  - `a7e3a14`. `comment_lines` uses `tokenize`'s `COMMENT` tokens and excludes the directives listed. The fuzzed constructed-module test checks it against counts known by construction, with directives, `#` in strings and `"""` in comments pinned; on `c724686` it reproduces the tests census, 467. At head: library 192, tests 467, scripts 12.
- [x] 1.2 Count docstring lines per file with `ast`, as the physical lines the docstring spans. In `SpectralSVR` a name is public when it carries no leading underscore and its package exports it, by re-export from an `__init__` or by definition in one; methods follow their class, nested definitions are internal, and a module docstring is public when the module exports a name. In `tests` and `scripts`, which export nothing, the leading underscore alone decides. Verify against constructed modules and export maps, and against the `tests` census (198 public, 54 internal). *Amended during apply:* the library split of 1229 against 123 cannot be reproduced by any reading either, and the export rule, decided by the user to follow the spec, moves unexported library names into the budget
  - `a7e3a14`. `docstring_lines` counts the lines of each docstring's content, the method that reproduces the tests census on `c724686` (198 public, 54 internal; counting the physical lines spanned gave 212). The export rule follows the spec; a fuzzed test checks the split over generated names and export sets for the library and the underscore-only rule. At head: library 989 public against 403 internal, tests 217 against 54, scripts 101 against 0.
- [x] 1.3 Verify the counter is blind to `#` inside string literals and to `"""` inside comments, by constructing both cases and asserting the count is unchanged
  - `a7e3a14`. Pinned examples of the constructed-module test: a `#` inside a string literal counts zero, and a comment containing `"""` counts as one comment and no docstring.

## 2. The budget

- [x] 2.1 Record the per-file budget seeded from 1.1 and 1.2, with internal docstring lines and comment lines as separate columns so a file's two debts are visible apart; verify the guard is green on arrival with no file over budget
  - `a7e3a14`. `tests/prose_budget.toml` records `comments` and `internal_docstrings` per file for the 40 files carrying either, seeded from the head; the repository checks pass on arrival.
- [x] 2.2 Fail a file that exceeds its budget, naming the file, the budget and the count; verify by adding a comment line to a budgeted file and confirming the guard names it
  - `a7e3a14`. Adding one comment to `scripts/run_benchmark.py` failed with `scripts/run_benchmark.py: comments 13 over its budget of 12`; reverted.
- [x] 2.3 Fail a file carrying counted prose with no budget entry, so an unlisted file is budgeted at zero rather than unbounded; verify with a new file carrying one comment
  - `a7e3a14`. A new `tests/_scratch_new.py` carrying one comment failed with `comments 1 over its budget of 0`; removed.
- [x] 2.4 Fail a budget that exceeds its file's actual count, so slack is reclaimed rather than banked -- the same shape as `test_no_exemption_is_stale`; verify by deleting a comment without lowering the budget and confirming the guard forces the edit
  - `a7e3a14`. Deleting one comment from the script without lowering its budget failed the slack check, naming the budget of 12 against the 11 the file carried; reverted.
- [x] 2.5 Verify the budget is per file rather than pooled: add a comment to one file, remove one from another, confirm the guard still fails
  - `a7e3a14`. Removing a comment from the script and adding one to `tests/conftest.py` failed both: the script's slack and `conftest.py`'s `comments 3 over its budget of 2`. `BudgetLedger` checks the same per-file holding against an independent model over sequences, and a pooled-total mutant fails it.

## 3. Docstring shape

- [x] 3.1 Record in design.md the content heuristic measured during apply and rejected by the user, with both measured sets, as the alternative rejected. *Replaces the original 3.1 to 3.4, which tuned that heuristic*
  - `9db16be`. design.md records the heuristic as the alternative rejected, with both measured sets: the first candidate caught both known-bad docstrings and fired on 18 of 243; the best tuned rule caught both and fired on none, fitted to two positives with three contracts one sentence below its threshold.
- [x] 3.2 Parse the docstring shape: a one-sentence summary, then numpydoc sections from the closed set (Parameters, Returns, Yields, Raises, Warns, Attributes, See Also, Examples), each entry's description at most one sentence, with terminators counted outside inline code and outside a closed abbreviation list, and Examples bodies not counted. Verify by fuzzing constructed docstrings of known conformance, with an abbreviation, a decimal inside inline code, a Notes section, a second summary sentence and a two-sentence entry pinned
  - `562d366`. `is_in_shape` and `section_entry_descriptions` parse numpy structure; ruff's numpy-convention `D` rules, excluding `D1`, supply the standard checks. A fuzzed test of constructed numpy docstrings pins an abbreviation, a decimal in inline code, a Notes section, a second summary sentence, a two-sentence entry, a headerless body and an Examples body; the fuzzer found Notes after Examples swallowed as example code, fixed by treating every underlined line as a header.
- [x] 3.3 Measure conformance over every docstring in the repository, public and internal, and record the count per area; seed `misshapen_docstrings` per file from it so the guard is green on arrival. Verify the two known-bad docstrings from `4e2b786`'s parent fall outside the shape
  - `562d366`. Misshapen at head: library 120 of 161, tests 37, scripts 7; `misshapen_docstrings` seeded for 47 files and the guard green on arrival. Both known-bad docstrings from `4e2b786`'s parent fall outside the shape. Measuring exposed that a re-exporting `__init__` exported nothing, moving `SpectralSVR/operators/__init__.py`'s two lines from internal to public once fixed.
- [x] 3.4 Refuse a misshapen docstring a file's budget does not count, naming the file; verify by adding a two-sentence summary to a conforming docstring and confirming the guard names its file, and that slack in the third column is reported like the other two
  - `562d366`. A second summary sentence in `Budget`'s docstring failed with `tests/_prose.py: misshapen_docstrings 1 over its budget of 0`; reshaping `SpectralSVR/operators/__init__.py`'s docstring without lowering its budget reported the slack. Both reverted. A fuzzed test over a temporary tree with the repository's convention separates ruff's verdict from the constraint's.
- [x] 3.5 Accept a conforming contract however much mathematics it states; verify with a docstring whose Parameters and Raises entries each state a bound with operators in one sentence
  - `562d366`. The pinned contract states `ns >= 6 * cutoff` in a Parameters entry and `ns < 6 * cutoff` in a Raises entry, each one sentence, and is in shape; the pinned derivation with the same mathematics across sentences is not.

## 4. Scenario coverage

- [x] 4.1 Mark each of the capability's scenarios with `@pytest.mark.scenario` as its task completes and remove its `IN_FLIGHT` entry in the same commit. The capability's scenarios, nine at proposal and ten after the shape revision, enter `EXEMPT` when this change is proposed, because it is parked at the stack tip with its artifacts written and no implementation, and a stated scenario with neither a test nor an entry leaves the suite red. Verify the stale-exemption test forces each removal rather than tolerating it
  - `a15ba5b`, `562d366`. Every scenario was named as its task completed and its in-flight entry removed in the same commit; the staleness check refused each entry once a test named it, which is what forced the removals.
- [x] 4.2 Fuzz the counting properties over generated sources rather than pinning one file: comment and docstring counts over constructed modules, and the public/internal split over generated names. Verify each property fails under a mutation of the behaviour it describes
  - `a7e3a14`. The constructed-module and public-split tests are fuzzed with edges pinned, and `BudgetLedger` drives budget sequences. Hand mutants each failed a test: directives counted, the export rule ignored, the underscore ignored, the over-budget check disabled, the slack check disabled, a pooled total, docstrings uncounted.
- [x] 4.3 Verify no scenario of this capability sits in `EXEMPT` at the end, so every entry added for this capability is discharged rather than archived as debt
  - None of the capability's ten scenarios sits in `EXEMPT`, and a test names each one.

## 5. Verification

- [x] 5.1 Wire the guard into `just check`; verify clean
  - The guard is `tests/test_prose_budget.py`, collected by `just check` with the rest of the suite: 400 passed and 2 failed, the pre-existing failures in `test_time_varying_scoring.py` that `condition-the-interpolating-fixture` fixes.
- [x] 5.2 Verify the guard adds no dependency: `tokenize` and `ast` are standard library, and `pyproject.toml` and `uv.lock` are untouched
  - *Amended during apply, at the user's direction:* the counter uses only `tokenize`, `ast`, `inspect` and `tomllib` from the standard library, but the shape rule runs ruff's numpy-convention rules, so `pyproject.toml` states that convention and the test extra declares ruff at the lint extra's pin; `uv.lock` gains that one dependency (`624f272`). No other dependency changed.
- [x] 5.3 Verify the guard itself obeys the rule it enforces -- its own comment and internal-docstring lines are budgeted like any other file, with no exemption for being the guard
  - `tests/_prose.py` and `tests/test_prose_budget.py` carry no comment lines, no internal docstring lines and no misshapen docstrings, and have no budget entry, so they are held to zero like any new file.
- [x] 5.4 Verify no reported number moves and nothing executable changes, by diff argument: this change adds a test and a data file and touches no library source
  - `git diff ccfbe18..HEAD --stat -- SpectralSVR scripts notebooks benchmarks` is empty: the change adds tests, a budget file and configuration, and moves no reported number.

## 6. Review fixes

*Added after the review, every finding in scope at the user's direction.* Each fix starts from the review's failing probe, which becomes a pinned case before the code changes.

- [ ] 6.1 Count a ruff violation anywhere within a docstring's span, not only on its first line; verify a fixture with D406, D407, D409 and D412 counts as misshapen, and that every ruff violation in the repository falls inside some counted docstring
- [ ] 6.2 Let an Examples section hold only doctest lines and their output; verify prose in Examples refuses the shape and a doctest-only section is accepted
- [ ] 6.3 Walk definitions under module-level `if`, `try` and `with`; verify a two-sentence docstring under each is counted
- [ ] 6.4 Count an unterminated trailing sentence; verify `Do x. And y` counts two and `Do x` counts one
- [ ] 6.5 Carry visibility down through class nesting; verify a method of a public nested class is public and one under an internal class is internal
- [ ] 6.6 Match machine directives case-insensitively, add `pylint:`, and exclude the coding cookie; verify each tool's documented directive form is not counted
- [ ] 6.7 Stop refusing an ellipsis and the common abbreviations: an ellipsis is not a terminator and the closed abbreviation list grows to the titles and Latin forms in use; verify `Wait... then go.` and `Ask Dr. Smith.` count one sentence
- [ ] 6.8 Admit Other Parameters, Receives and Methods as entry lists; verify each is accepted with one-sentence entries and refused with a two-sentence entry, and that Warnings and References are refused
- [ ] 6.9 Validate the budget file: unknown keys and negative values raise a named error stating the file and the key; verify both
- [ ] 6.10 Make the export map follow absolute imports of the package, `__all__`, star imports and imports under a module-level `if`; verify each against a temporary package
- [ ] 6.11 Ignore ruff entries that are not docstring violations, such as syntax errors; verify a file with a syntax error does not mark a docstring misshapen
- [ ] 6.12 Anchor the rules the review found held only by consistency, and kill its ten surviving mutants: the closed set as the spec's literal list in the test, a hand-counted sentence corpus, the ruff join against ruff's own JSON over a fixture, a directive table, the export map against Python's own `dir()` of a temporary package, and `BudgetLedger` checking which files and columns are reported with an honest oracle marker; verify each of the ten mutants and the earlier seven fail a test with the repository checks deselected
- [ ] 6.13 Reshape `tests/test_scenario_coverage.py`'s module docstring, a touched file, to the shape, lowering its budget
- [ ] 6.14 Run `just check`, and verify no explanatory prose comment was added and every touched docstring is in shape

