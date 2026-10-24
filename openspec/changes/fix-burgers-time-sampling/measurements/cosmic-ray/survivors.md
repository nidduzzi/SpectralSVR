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
