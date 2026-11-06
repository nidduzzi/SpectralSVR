# Surviving mutants of fix-real-derivative-at-nyquist

## How the mutants were run

Cosmic-ray ran on commit 11f5d06, in a clean worktree with its own environment, inside the resource wrapper. Only the line ranges this change touched were kept. Each mutant ran against the tests that cover its line, using the covering-test map from the archived consolidate-the-test-suite change. `run.sh` and the four `.toml` files hold the commands. The `*-report.txt` files hold cosmic-ray's own output.

| module | mutants | killed by covering tests | survived |
| --- | --- | --- | --- |
| `SpectralSVR/basis/FourierBasis.py` | 379 | 340 | 39 |
| `SpectralSVR/basis/__base.py` | 4 | 0 | 4 |
| `SpectralSVR/basis/_nufft.py` | 0 | | |
| `SpectralSVR/operators/dataset.py` | 0 | | |

`_nufft.py` line 60 is `return values`. `dataset.py` lines 131 to 134 pass keyword arguments through. No cosmic-ray operator applies to either.

Covering-test selection misses a test that reaches a line only through a short-circuited condition. So each of the 43 survivors was then applied with `cosmic-ray apply` and run against the whole suite with `-x` (`rerun_survivors.py`). The unmutated baseline gave 468 passed in 45 s. `whole_suite_rerun_11f5d06.jsonl` records each mutant's diff, exit code and last line.

- 1 mutant was killed by the whole suite.
- 42 mutants survived it.

## Verdicts

Line numbers are those of 11f5d06.

### Killed by the whole suite

- **531, `and nufft_available()` replaced by `or`.** Three tests failed, among them `test_a_field_reads_as_its_closed_form_off_and_on_its_grid`. The covering-test map missed them because the line sits inside a short-circuited condition.

### Gaps closed by this change

`check_kills.py` applies each of these mutants to commit ba094bf. `kill_check_ba094bf.txt` records the result.

- **530, eight operator mutants of `res.step * read_modes`.** This was a defect, not only a gap. d5106a5 made the off-grid inverse transform count the split modes when it chooses the approximate path. The design says the switch counts the stored modes. ba094bf restores the stored count. The switch test in `test_nufft.py` now reads both off-grid evaluation and the inverse transform on a partial window. It kills all eight mutants of the corrected expression, and the split count as well.
- **360, `ord: int = 1` replaced by 0 or 2, and `complex_funcs: bool = False` replaced by True.** These are the defaults of the static `derivative_eigenvalues`. Every library call passes both arguments. The Burgers solver relies on the `complex_funcs` default, but its de-aliasing mask removes the boundary eigenvalue. `test_derivatives_and_integrals_follow_the_closed_form` now compares the default call with `2 pi i k / L`, the boundary entry zeroed on an even count. That test kills all three mutants.
- **529, `func == "inverse"` replaced by `func <= "inverse"`.** "forward" sorts before "inverse", so the forward transform would send sample values through the NUFFT. Every forward transform in the suite ran under the default 256 MB budget. a094dfc runs the closed-form test of the explicit forward leg with `memory_budget_mb=0.0`, which kills the mutant.

### Equivalent

- **364, `modes % 2 == 0` replaced by `<= 0`.** For any integer, `% 2` gives 0 or 1, so the two comparisons agree.
- **365, `ord % 2 == 1` replaced by `>= 1`.** The same argument applies.
- **369, `@staticmethod` removed from `split_boundary_modes`.** Every call goes through the class (`FourierBasis.split_boundary_modes`, `cls.split_boundary_modes`) and passes the coefficients explicitly.
- **372, eleven mutants of `m + 1 - m % 2`.** Each mutant returns a different size, but in every case the read value is the same up to rounding:
  - an even axis grows to `m + 1`, `m + 2` or `2m`, which splits the stored boundary coefficient between the interior modes `+m/2` and `-m/2`;
  - an odd axis keeps `m` or grows by zero-padding;
  - wherever the grown size is even, its own boundary coefficient is zero, so it reads the same as either wave.

  Scaling divides by the stored count, never the read count, so no mutant changes the scale. The approximate switch now counts stored modes too (ba094bf), so no mutant changes that decision either.
- **374, `sample_axes - 1` replaced by `sample_axes >> 1`.** Every call passes `sample_axes` 1 or 2. For those values, `1 - 1 = 1 >> 1 = 0` and `2 - 1 = 2 >> 1 = 1`.
- **491 and 557, private `complex_funcs: bool = False` replaced by True** on `_raw_transform` and `_ndim_transform`. Every inverse call passes `complex_funcs` explicitly. Forward calls take the default, and the forward path ignores the flag.
- **524 and 529, `func == "inverse"` replaced by `func >= "inverse"`.** "forward" >= "inverse" is False, so these mutants never change a branch.
- **524, `func == "inverse"` replaced by `func <= "inverse"`.** This mutant splits the forward transform's input too. On the native periodic grid the halves at sample positions `+N/2` and `-N/2` meet `exp(+i pi j) = exp(-i pi j)` at every node `j`, so the result is identical. Every forward transform the library makes covers its whole domain at its own sample count. A probe of `to_time_dependent(nt).to_time_independent()` at nt 4, 5, 6 and 8 gave bitwise-identical coefficients with and without the mutant. The mutant is observable only when a forward transform runs on a partial window, which no library caller and no test does. It is recorded as equivalent over every reachable forward transform.
- **`__base.py` 565 and 726, the abstract signatures' defaults.** These are `complex_funcs` on `evaluate_time_dependent_at`, and `ord` and `complex_funcs` on `derivative_eigenvalues`. Python binds the defaults of the override that runs, never those of the abstract declaration. `Basis.derivative_eigenvalues` returns `None` whatever its arguments.

### Gaps in code this change did not alter

These lines sit inside a function the change touched, but d5106a5 left them unchanged. Each mutant survived the whole suite. They are recorded here and deferred to their own change.

- **673, 674, 675, `cdim - 1` replaced by `cdim >> 1`** in `inv_transform`. The two agree for `cdim` 1 and 2, and differ only for a field with three spatial axes. No test reads one through `inv_transform`.
- **686, `dim - 1` replaced by `dim >> 1`, `dim ^ 1` or `dim % 1`** in `_diff_multiplier`. `^ 1` and `% 1` change `dim` 2. That case is the second spatial axis of a time-dependent field, so differentiating a time-dependent field along a second spatial axis is untested. `>> 1` differs only from `dim` 3.

## Outcome

Of the 383 mutants on the changed lines:

- 340 were killed by the covering tests;
- 1 was killed by the whole suite;
- 12 were killed by tests added in ba094bf and a094dfc;
- 24 are equivalent;
- 6 sit in unchanged lines inside touched functions and are deferred.
