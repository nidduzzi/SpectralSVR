# Survivors over the manufactured branch of `SpectralSVR/problems/burgers.py`

Run at `ca83e0b` over lines 141 to 162, the dispatch condition and the manufactured branch: 19 mutants, each against the tests that cover its line (the archived consolidation's `selected_pytest.sh`); 5 survive. Each survivor was then applied by hand in the mutation worktree and run against the whole suite (`pytest -n 6 tests`, 459 tests, `--hypothesis-seed=0`).

The changed lines, 160 to 162, carry two mutants, both `not` inserted in `if time_dependent_coeff`; both are killed. The two `to_time_dependent(nt=nt)` calls offer no operator a mutant.

| line | mutant | whole suite | verdict | reason |
| --- | --- | --- | --- | --- |
| 145 | `and f == "random"` to `or f == "random"` | killed | selection miss | `test_numerical_branch_rejects_a_manufactured_forcing` kills it. Coverage records line 145 only for calls that pass `u0 == "random"` first, so tests with a numeric `u0` never reach it and are not selected |
| 143 | `u0 == "random"` to `<=` | survives | gap | `u0="abc", f="random"` is silently manufactured instead of refused; no test passes a string other than "random". Recorded by the consolidation audit (`problems_burgers.md`, line 143), deferred to close-the-mutation-gaps |
| 143 | `u0 == "random"` to `>=` | survives | gap | same, for a string sorting after "random" |
| 145 | `f == "random"` to `<=` | survives | gap | same for `f` with `u0="random"`; audit line 145 |
| 145 | `f == "random"` to `>=` | survives | gap | same |

None of the gaps is in a line this change touches.

# Survivors over the slice-position change

Run at `3917c0a`, each mutant against the tests that cover its line. Covered ranges: `SlicePositions` and `interpolate_tensor` in `SpectralSVR/utils/__init__.py` (rows 634 to 677, 130 mutants); the routed reads in `SpectralSVR/basis/__base.py`, which are `get_values`, resizing in time and `to_time_independent` (rows 300 to 305, 905 to 913 and 1088 to 1092, 20 mutants); and the time-dependent `evaluate` and `evaluate_time_dependent_at` in `SpectralSVR/basis/FourierBasis.py` (rows 139 to 145 and 196 to 205, 31 mutants). Of 181 mutants, 5 survived. Each was then applied by hand and run against the whole suite, at `--hypothesis-seed=0`.

| file:row | mutant | whole suite at 3917c0a | verdict | after |
| --- | --- | --- | --- | --- |
| utils:634 | `frozen=True` to `False` | survives (the decorator runs at import, so every test covers it) | gap | killed at `008890c`: the refusal test pins that moving a position past the last slice raises `FrozenInstanceError`; the whole suite then fails 1 of 463 |
| __base:908 | resize target times on `domain[1]` | survives (463 passed) | gap | killed at `cada220`: `resize_through_time` compares coefficients with an explicit lerp, and `set_domain` separates the axes' intervals |
| __base:908 | resize target times on `domain[-1]` | survives | gap | same |
| __base:911 | resize positions on `domain[1]` | survives | gap | same |
| __base:911 | resize positions on `domain[-1]` | survives | gap | same |

All 31 FourierBasis mutants are killed, among them every mutant of the positions' lower, upper and weight in `evaluate_time_dependent_at`.
