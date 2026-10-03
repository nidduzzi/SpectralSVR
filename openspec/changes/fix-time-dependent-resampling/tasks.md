## 1. Pin the defects

- [x] 1.1 Register the change's two scenarios as in flight in `tests/test_scenario_coverage.py`; verify the scenario guard passes with them
  - Landed with the proposal in `c005dfd`; the scenario guard passes (5 passed).
- [x] 1.2 Write the fuzzed conversion test first: slice counts drawn below, at and above the time axis's mode count, with modes `(3, 2)` at `nt=2`, each of those three relations and the default pinned; at the slice times the converted field must match the source to rounding, between them within `Δt² / 8 · max |∂²u/∂t²|`; verify it fails on the current code, on the crash and on the placement
  - `test_to_time_dependent_reads_as_its_source_at_every_time` in `tests/basis/test_coeff_anchors.py`, fuzzed over modes (2..8, 2..6), slices None or 2..12, span 0.5..4, phase, probe times; pinned: modes (3, 2) at nt=2, below (8,4)/5, at (6,4)/6, above (4,3)/11, default (8,8). On the parent it failed on both defects: `RuntimeError: shape '[1, 3, 2]' is invalid for input of size 4` on three examples, and a slice-time error of 1.17 on two, the default among them.
- [x] 1.3 Measure the agreement at the slice times over the drawn range before setting the rounding tolerance, and record the measurement here
  - 5000 seeded draws over the test's ranges, `c005dfd` plus the fix, complex128 coefficients (the test builds its fields in double): worst slice-time error 8.2e-15; worst between-slice error 0.992 of the interpolation bound, so the bound is tight. `SLICE_TIME_ROUNDING` set to 1e-12, about 120 times the measured worst.

## 2. Fix

- [x] 2.1 Make `to_time_dependent` evaluate the source at the inclusive slice times on each spatial axis's periodic grid and reshape to `(len, nt, *modes[1:])`, refusing fewer than two slices with a named error; verify the new test passes and the refusal raises naming the count
  - The conversion asks `inv_transform` for `ClosedUniform` nodes along time and the basis's own scheme along each spatial axis, transforms with the same spatial schemes, and unflattens to `(len, nt, *modes[1:])`; fewer than two slices raises `ValueError` ending `at least two slices; got {nt}`, checked by `test_to_time_dependent_refuses_fewer_than_two_slices` (slices -3..1, 1 and 0 pinned). The new tests pass.
- [x] 2.2 Rewrite `test_to_time_dependent_slice_spectra_match_analytic` to expect slice `j` at `j / (nt - 1)` with its tolerance unchanged; verify it fails on the old placement and passes on the new
  - Slice `j` is now expected at `linspace(0, 1, nt)[j]`, tolerance unchanged; the section banner and inline comments around it went. It fails on the old placement at slice 3 and on the restored half-open mutant alone; it passes on the fix.
- [x] 2.3 Name the two scenarios on the tests that check them and remove their in-flight entries; verify the scenario guards pass
  - Both scenarios named on `test_to_time_dependent_reads_as_its_source_at_every_time`; `tests/test_scenario_coverage.py` is back to its parent's content and its guards pass.

## 3. Verification

- [x] 3.1 Run `just check`; verify the suite passes in full, naming any test whose expectation moved, and record the wall time against the parent
  - Rerun after section 4: `just check` green, ruff, mypy and ty clean, 427 passed in 91.4 s (95 s wall), against 424 in about 90 s on the parent. Three tests' expectations moved, each because it encoded a misplacement: `test_to_time_dependent_slice_spectra_match_analytic` (2.2), `test_to_time_independent_values_match_analytic` (4.5) and `test_to_time_dependent_preserves_values` (4.6).
  - First run: 1 failed, 425 passed in 98 s. `test_to_time_dependent_preserves_values` failed because `to_time_independent` carries the mirror defect (see the proposal), which led to section 4. Rerun in 4.8.
- [x] 3.2 Verify no benchmark row moves: the benchmark's time-dependent fields come from the Burgers generator, untouched here, so a diff argument suffices
  - The only other `to_time_dependent` in the library is a local function inside `SpectralSVR/problems/burgers.py`'s generator, which does not call the method; nothing on the benchmark's path reaches `Basis.to_time_dependent`, so no row moves.
- [x] 3.3 Hand-mutate the fix (half-open placement restored, the old reshape restored, the two-slice refusal removed) and verify each fails a test
  - All killed against `tests/basis/test_coeff_anchors.py`: half-open placement restored, old reshape restored, two-slice refusal removed, default slice count taken from a spatial axis; the half-open mutant is also killed by the rewritten anchor alone. Rerun with section 4 in 4.8.
- [x] 3.4 Verify no explanatory prose comment was added and every touched docstring is in shape, lowering the prose budget where touched prose went
  - No comment was added in the diff. No docstring was touched. `tests/basis/test_coeff_anchors.py` dropped from 58 to 51 comments, and its budget was lowered to match. Rerun in 4.8.

## 4. The inverse conversion

*Added after the gate's first run; the user decided to fix it here by interpolation.*

- [x] 4.1 Register the two new scenarios as in flight; verify the scenario guard passes
  - Two `IN_FLIGHT` entries landed with the artifact revision in `c246019`; the scenario guard passed there.
- [x] 4.2 Write the fuzzed tests first: a converted time-dependent field reads, at each node of its time-independent result, as the time-dependent field at that node's time; a field periodic in time round-trips within `Δt² / 8 · max |∂²u/∂t²|`, with slice counts below, at and above the time axis's mode count pinned; verify both fail on the current code
  - `test_to_time_independent_reads_as_its_time_dependent_source_at_its_nodes` (oracle_independent: `numpy.interp` of drawn slice values; slices 2..12, spatial modes 2..6, span, seed; 2, 8 and 12 slices pinned). The round trip joins the forward test, renamed `test_to_time_dependent_and_back_read_as_the_source`, so it inherits the below, at, above and default pins. On the code before 4.4: node error 1.86, and a round trip of 0.89 against a bound of 0.31.
- [x] 4.3 Measure the node agreement and the round-trip error against the bound over the drawn ranges, record them here, and set the rounding tolerance above what was measured
  - 5000 seeded draws each, complex128: worst node error against `numpy.interp` 6.9e-15; worst round trip 0.993 of the bound, exceeding it by at most 8.9e-16, only where the bound is zero. One constant, `DOUBLE_PRECISION_ROUNDING = 1e-12`, replaces both slice-time and node tolerances, about 120 times the worst of either. The complex64 consistency test (4.6) was measured separately over modes 2..8 by 300 seeds: forward worst 4.17e-6 absolute; round trip at most 0.714 of its bound.
- [x] 4.4 Make `to_time_independent` read the slices at the periodic times by linear interpolation and transform; verify the new tests pass
  - `to_time_independent` inverse-transforms each slice on the basis's spatial schemes, reads the slices at the time axis's default-scheme nodes (periodic for Fourier) through `fractional_slice_index` and `interpolate_time_tensor`, and transforms with that time scheme and the spatial ones. The time axis keeps `time_size` modes. The new tests pass.
- [x] 4.5 Rewrite `test_to_time_independent_values_match_analytic` to expect each half-open node to hold the slices' linear interpolation at its time, computed with `numpy.interp`, and state the oracle class honestly; verify it fails on the closed-time read and passes on the fix
  - Expected values are now `numpy.interp` of the closed-slice `g` at the half-open nodes, times the spatial factor, with the same `1e-3 * amp` tolerance; the marker went from oracle_analytic to oracle_independent, since `numpy.interp` carries the expectation. Its prose comments went. It kills the restored closed-time read alone.
- [x] 4.6 Rewrite `test_to_time_dependent_preserves_values` to compare each representation with the source at its own grid's coordinates, and the round trip within the interpolation bound, stating the oracle class honestly; verify it fails with either method's misplacement restored
  - The forward half compares the converted field with the source evaluated at the converted field's own grid. The round trip compares with the source at the result's own nodes, within `Δt²/8` times the spectral bound `Σ |c_k| (2π k_t)² / m²` on `max |u_tt|`. It stays oracle_consistency, since the source's own evaluation is the reference. The old `atol=1e-6` belonged to a comparison of misaligned grids; the new forward comparison measured 4.17e-6, so it is held at an absolute `SINGLE_PRECISION_ROUNDING = 1e-5`. It kills either method's misplacement restored alone.
- [x] 4.7 Name the two scenarios on the tests that check them and remove their in-flight entries; verify the scenario guards pass
  - `A round trip stays within interpolation error` is named on the forward test, and the node scenario on the new inverse test. `tests/test_scenario_coverage.py` is back to its content at `734d7f9`. Guards pass.
- [x] 4.8 Rerun 3.1, 3.3 with the closed-time read in `to_time_independent` restored as a further mutant, and 3.4
  - 3.1 above. 3.3 against both basis modules: all six mutants killed (half-open placement, old reshape, refusal removed, default from a spatial axis, inverse closed-time read, inverse slice index off by one). 3.4: no comment or docstring added; comment budgets lowered for `__base.py` 20 to 19, `test_coeff_anchors.py` 51 to 44, and `test_fourier_basis.py` 77 to 72.

## 5. Review fixes

*Added after the review; every finding in scope at the user's direction. Folding the conversions into `BasisLifecycle` stays with `consolidate-the-test-suite`.*

- [ ] 5.1 Register the two new scenarios as in flight; verify the scenario guard passes
- [ ] 5.2 Construct each conversion's result with the class constructor and a sampling of the right arity (time scheme dropped forward, prepended back); verify a per-axis sampling's round trip matches the default-sampling one within the interpolation bound, and that restoring the copied sampling fails a test
- [ ] 5.3 Return a copy for a time-dependent field converted with no slice count or its own, and raise naming both counts otherwise; verify both branches
- [ ] 5.4 Refuse a non-integer or boolean slice count with `TypeError`, and an empty basis and a field without a spatial axis with `ValueError`; verify each raises its named error and the refusal test covers every path
- [ ] 5.5 Sample the inverse's time axis on `PeriodicUniform()` explicitly, and resolve spatial schemes through a `spatial_sampling` property used by both conversions and `get_values`; verify the suite passes
- [ ] 5.6 State each conversion's contract in a one-line numpy docstring with a `Raises` section; verify the docstring shape guard passes and no free prose is added
- [ ] 5.7 Draw the time interval's start in both fuzzed tests and pin `start=1.0, span=2.0`; verify the forward and inverse start-ignoring mutants fail
- [ ] 5.8 Draw a non-FFT `PeriodicUniform` and a non-default strategy where the API admits one, to separate the mutants that drop spatial sampling, domain or strategy; record any that stay equivalent and why
- [ ] 5.9 Measure both rewritten anchors' errors over their ranges, record them, and tighten their tolerances above what was measured
- [ ] 5.10 Take oracle times from `torch.linspace` and `half_open_grid` rather than the library's sampling schemes; verify the tests pass
- [ ] 5.11 Replace `test_to_time_dependent_preserves_values`'s reference with an independent numpy point evaluation of the spectrum, measured over its full seed range before bounding, and mark its oracle honestly
- [ ] 5.12 Draw precision (complex64, complex128), batch size above one and complex-valued fields in the conversion tests, pinning complex64 and a batch of three; verify they pass at the measured tolerances
- [ ] 5.13 Delete `test_to_time_dependent_and_back` and `test_to_time_dependent_default_nt`, which the fuzzed tests subsume; verify the default and round trip stay pinned there
- [ ] 5.14 Cut `tests/basis/test_coeff_anchors.py`'s module docstring to one sentence that holds for the module
- [ ] 5.15 Name the new scenarios and remove their in-flight entries; run `just check`, rerun the hand mutants with the new ones, and the prose audit

