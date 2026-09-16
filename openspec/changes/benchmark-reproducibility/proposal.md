# Make the comparison table reproducible and honestly reported

## Why

The comparison table is the paper's headline result and is currently a single draw with no provenance. `results/bench_results.csv` carries one row per dataset and operator, produced at seed 0, with no record of what produced it. Four gaps, each of which a reviewer can raise.

**One seed.** Every figure is a single sample. No spread is reported, so a gap between two operators cannot be distinguished from the variation between draws. `scripts/run_benchmark.py` accepts `--seed` but nothing aggregates across seeds.

**No provenance.** The table cannot be tied to the code or hardware that produced it. That matters here more than usual: the repository backs a paper, and both the coefficient precision and the device change the numbers. A table that cannot be traced to a revision cannot be defended.

**A metric a reader will misread.** Neural-operator papers report per-sample relative L2, averaged over the test set. The table reports `rrse`, a single global `sqrt(SSE/SST)` taken over all elements at once. These are different statistics, and a reader placing this table beside a DeepONet or FNO table will assume the former. The per-sample form belongs alongside the global one, not instead of it.

**Times that are not comparable.** `fit_seconds` brackets `fit()` alone. For SpectralSVR that is a direct solve; for DeepONet it is a fixed ten-thousand-iteration training loop, possibly on a different device. Two numbers in one column that measure different things invite exactly the comparison they cannot support.

## What Changes

- The benchmark runs over a fixed set of seeds and reports a centre and a spread for every metric, so a difference can be read against the variation.
- Each run records what produced it: the device **each stage** ran on, coefficient precision, torch and deepxde versions, and the code revision. A single device column would misreport today's runs, which use two — `Burgers.generate:62` integrates on `cuda:0` via `resolve_device()` while `MultiRegression.__init__` at `model/__base.py:49` hardcodes `cpu` — both verified still true on this branch.

*Measured while implementing `dtype-seam`, and recorded here because it bears on what this change must do.* Six consecutive runs of identical code in the same tree vary by up to `2.51e-05` relative — the harness is not run-to-run deterministic, so a single-seed table cannot be reproduced exactly even on the machine that made it. Pinning `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 cuts that to `6.15e-07`, which points at multi-threaded reductions in the solve. These figures are from one machine and should be re-measured under this change's own protocol rather than treated as constants.

That noise does **not** confound the seed spread this change reports. Between-seed variation over seeds 0-4 runs `26,010x` the noise floor on the antiderivative row, `12,291x` on Burgers MMS, `96,434x` on Burgers numerical and `180,953x` on the GRF row — `mse` alone moves 242% and 454% across seeds. The single-draw problem this change exists to fix is four to five orders of magnitude larger than the measurement noise, so reporting seed spread without separating run-to-run noise is sound; it is worth stating that it was checked rather than assumed.
- Per-sample relative L2 is reported alongside the existing global `rrse`, with both named unambiguously so neither is mistaken for the other.
- The timing column is documented as measuring different work for different operators, or split so it is not read as like-for-like.
- A `just` recipe regenerates the table in one command, and the comparison notebook reads the committed results instead of recomputing them.

**Not in scope**, each for its own reasons: hyperparameter tuning, which needs a protocol and a change of its own; new baselines such as FNO; new datasets beyond the one the forcing change adds; and any change to the metric definitions themselves, which would move numbers for reasons unrelated to reproducibility.

## Capabilities

### Modified Capabilities
- `operator-comparison`: the table gains a reproducibility contract — repeated seeds, recorded provenance, and an unambiguous error statistic. Two of these are **modifications rather than additions**, because `operator-harness-spec` specified this capability after this change was drafted:
  - *The table states what it does and does not establish* is rewritten. As landed it requires documentation saying results come from a single seed; this change makes that false, so it supersedes the requirement rather than contradicting it from a second one. The tuning caveat stays, because tuning is still uncontrolled.
  - *A row says how its forcing was obtained* is rewritten so "the rows already reported are unchanged" is checked by comparing each metric's centre and spread over the same seed set. Once a row is a distribution rather than a draw, comparing single values would fail for reasons unrelated to the change being checked.

  **Ordering dependency.** The second modification targets a requirement that
  `burgers-forced-benchmark` introduces, so it only resolves once that change is
  archived and synced. `openspec validate` says so today:
  `Archive would refuse this delta: operator-comparison MODIFIED failed for
  header "### Requirement: A row says how its forcing was obtained" - not found`.
  The stack order already satisfies this — that change sits directly beneath —
  but archiving out of order would fail rather than silently drop the edit.

  A planned requirement about the timing column was **dropped**: the landed *Fit times are qualified* scenario already contracts the same behaviour, in almost the same words, and `test_the_benchmark_script_states_what_its_table_establishes` already covers it.

## Impact

- `scripts/run_benchmark.py`: seed aggregation, provenance capture, the additional metric column.
- `SpectralSVR/operators/benchmark.py`: per-sample relative L2 alongside the existing suite, without altering the existing metric definitions.
- `Justfile`: a `bench` recipe.
- `notebooks/comparisons/deeponet_vs_spectralsvr.ipynb`: read the committed table rather than recomputing it.
- `results/bench_results.csv`: regenerated, with more columns and one row per dataset, operator and seed, or an aggregate with spread.
- Depends on `align-spectral-residual-with-solver` (it moves the Burgers MMS row, and the multi-seed table should be generated after that), `operator-harness-spec` (this extends the capability that one states), and `time-dependent-forcing-in-numerical-burgers` (its forced dataset belongs in the multi-seed run rather than being appended afterwards).
