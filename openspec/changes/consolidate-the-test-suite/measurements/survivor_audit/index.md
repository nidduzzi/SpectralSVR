# Survivor audit

Population: the mutants that survived the final phase-1 to phase-4 cosmic-ray runs, one per module, operator and source line (2776). Each was rerun against the whole suite of 455 tests at `30b88d4` (`cosmic-ray/prepare_survivor_audit.py`, sessions `phase5audit-*`, Hypothesis seed 0, one mutant at a time under a 12 GiB, 400% CPU and 512-task scope). 976 survived. Each module's table lists them by source line, in this directory; `cosmic-ray/list_audit_survivors.py` regenerates the raw rows.

Verdicts are a reading of the source and the tests, not proofs. Equivalent: no input tells the mutant from the original. Dead: only an unreachable branch or an unsupported input does. Gap: an input exists that no test uses. Deferred: touches a known defect the consolidation does not pin.

## Totals (mutants)

| verdict | mutants | share |
| --- | --- | --- |
| gap | 589 | 60% |
| equivalent | 320 | 33% |
| dead | 61 | 6% |
| unsure | 4 | under 1% |
| deferred | 2 | under 1% |

| module | survivors | gap | equivalent | dead | other |
| --- | --- | --- | --- | --- | --- |
| basis/_plot.py | 191 | 166 | 16 | 9 | |
| model/LSSVR.py | 107 | 96 | 5 | 6 | |
| basis/FourierBasis.py | 101 | 49 | 42 | 7 | 3 unsure |
| basis/__base.py | 99 | 63 | 34 | 2 | |
| utils/__init__.py | 98 | 37 | 50 | 10 | 1 unsure |
| problems/burgers.py | 169 | 66 | 94 | 9 | |
| problems/exact.py | 45 | 44 | 0 | 1 | |
| model/__base.py | 40 | 28 | 7 | 5 | |
| model/SpectralSVR.py | 22 | 11 | 9 | 2 | |
| operators/deeponet.py | 23 | 2 | 17 | 4 | |
| problems/antiderivative.py | 9 | 5 | 2 | 2 | |
| problems/__base.py | 5 | 3 | 2 | 0 | |
| operators/benchmark.py | 2 | 2 | 0 | 0 | |
| small modules (config, sampling, domain, _nufft, FNN, dataset, operators/__base) | 65 | 17 | 42 | 4 | 2 deferred |

A mutant is not a missing test: one missing test often kills a whole family (the kernel arithmetic of LSSVR, the 11 operator swaps of one expression). The gap count over-reads the work by a large factor.

## Where the gaps cluster

- `basis/_plot.py`: the plot tests draw and assert nothing about the result, on one-function bases only. 166 of its 191.
- `model/LSSVR.py`: the poly, tri, frob and max kernel arithmetic is never pinned (the independent KKT test covers linear and rbf); the batched kernel path, the defaults, `y_indicies` and `load`'s type check.
- `problems/exact.py`: the residual checks accept any member of the manufactured-solution family, and no test compares a builder with a hand value or a shape. 44 of 45.
- `problems/burgers.py`: input validation, defaults, the de-aliasing mask (coefficients above it are 3.5e-17 with the mask and 0.25 without it on a test input) and the manufactured time-dependent output never compared with the field's values.
- `basis/__base.py` and `FourierBasis.py`: an even time length for the time derivative, values of spatial `grad` on a time-dependent field, forward `transform` with per-axis schemes or `res`, off-domain `res` windows against a dense reference, the forced-approximate forward transform, `read_a_selection_of_functions` pins `(start=1, count=1)` and `(2, 2)` (the pinned `(1, 2)` cannot tell `|`, `^` and `+` apart), and the two evaluation modes, which agree to about 1.4e-6 in float32 and are compared by shape only.
- `utils/__init__.py`: ETDRK4 with a start time other than zero and with complex eigenvalues, the complex weight node count, a constant column in the scaler, NaN through `transform`, one-dimensional `X`.
- `model/SpectralSVR.py` and `model/__base.py`: `test()` never asserts metric values, so the time-dependent grouping and the complex branch are untested; `forward` with a domain on a non-time-dependent model; one-dimensional `X`.
- The 17 gaps of the small modules: see `small_modules.md`.

## Decisions for the user

- Error-message text: 71 of burgers' 94 equivalent mutants only change a message. They are equivalent if no message is contract, gaps if any is.
- Batch size: 35 `LSSVR._batched_K` mutants change the block size without changing the output; they are counted as gaps and would move to equivalent if output-invariance counts.
- Defaults: changes to undocumented defaults (`inverse` epochs and gain, DeepONet iterations and seed, plot defaults, `derivative_eigenvalues` order) are counted equivalent or unsure; documented ones (`d=3`, `C=1.0`) are gaps.
- Plotting: whether `_plot.py` earns assertions beyond smoke.
- `problems/burgers.py` line 644 touches the deferred `ClosedUniform` defect and is marked gap; it may belong on the deferred list.

## What this does not establish

- One mutant stands for every mutant of the same operator on the same line; a line with a mix of operators may hide a different fate for a sibling that was not run.
- Equivalent and dead verdicts rest mostly on reading the code; the agents ran a small check for some and say so in the tables. A sample read by hand found them sound.
- The audit is non-regression plus a view of what the suite misses. It does not claim the old suite killed these either: the before runs were judged by changed tests only.
