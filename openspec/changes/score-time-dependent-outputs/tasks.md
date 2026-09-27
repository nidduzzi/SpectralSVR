## Execution order

```
1 scenarios as debt ── 2 paired evaluation ── 3 targets and forward ── 4 the forced row ── 5 verification
```

A chain. In sections 2 to 4, every property is written as a test that fails
against the code before the code that satisfies it is written. Fuzz rather than
pin: edges are pinned as `@example`s on the fuzzed tests. Where state carries
across calls, model it with a `RuleBasedStateMachine`. Measure every tolerance
before asserting it, and record the measurement here.

## 1. Scenarios as debt

- [x] 1.1 Add `IN_FLIGHT` entries for the five scenarios the delta states. Verify `test_every_stated_scenario_is_covered_or_exempt` fails without them and passes with them
  - Without the entries, the guard failed naming the 5 scenarios; with them, 3 passed.

## 2. Paired evaluation

- [x] 2.1 Invariant, `oracle_invariant`: a time-varying field evaluated at paired rows `(t_i, x_j)` equals the grid form's `out[:, i, j]`, for every `i` and `j`. Fuzz over the number of functions, slice count, mode count (even and odd) and device. Pin `t = t0`, `t = t1`, `x = 0`, `x = L`, a single time slice, and the smallest mode counts. Measure the gap before setting its bound. Verify it fails first
  - `test_paired_evaluation_equals_the_grid_form_at_every_pair`. It failed first with `AttributeError`. **Amended:** bit-identity does not hold, and cannot be asserted. Paired evaluation sums per slice at every row, repeated `x` included, while the grid sums at the unique points, so the matmul shapes differ and so does the rounding. The first failure was 2.4e-8 in an imaginary part. Measured over 300 trials on both devices: worst relative gap 1.8e-7. Bound `ROUNDOFF_RTOL = 1e-6` relative to the field's largest value. Pinned: a single slice at `t = 0` and `t = 1`; `t0`, `t1` and midway with an even mode count; `t1` and midway with an odd one; `x = 0` and `x = L` through the space axis `[0, 0.3, 1]`.
- [x] 2.2 Analytic anchor, `oracle_analytic`: a field manufactured from known values on the spatial grid at every slice gives those values exactly at slice times. Midway between two slices it gives their mean, by the stated linear convention. Fuzz as 2.1, with the same pins. Verify it fails first
  - `test_paired_evaluation_reads_slices_exactly_and_lerps_between_them`, `oracle_analytic`. It uses values manufactured on the spatial grid at every slice. Measured worst relative gap: 2.3e-7 at slices, 1.73e-7 midway against the mean of the neighbouring slices. Same bound. Pinned: a single slice with one mode, two slices with two modes, and 12 slices with 11 modes.
- [x] 2.3 A row carrying only spatial coordinates is refused for a time-varying field, with a message naming the `(t, x)` layout. Fuzz over spatial dimension and row count. Verify it fails first
  - `test_a_point_without_a_time_in_the_interval_is_refused`, fuzzed over spatial dimension and point count. **Amended:** it also refuses a time outside `[t0, t1]`. The plan missed that invalid state, in which a negative index would silently wrap to the far end of the time axis.
- [x] 2.4 Implement paired evaluation beside the grid form, per the design. Verify 2.1 to 2.3 pass, and that the grid form's existing tests pass unchanged
  - `FourierBasis.evaluate_time_dependent_at(coeff, points, domain)`, declared abstract on `Basis` with a one-line contract. It evaluates the spatial basis at each row, sums per slice, then applies `torch.lerp` between each row's two slices, with the grid form's weight and scaling expressions. The index map is now one function, `fractional_slice_index`, used by both forms so they cannot drift. Removed where touched: the grid branch's two prose comments and the `## Fourier basis` section comment. 137 basis tests pass, the grid form's included.

## 3. Targets and forward

- [x] 3.1 Cover `Targets of a time-varying output are read at a time and a place`: `targets` on a time-varying output equals paired evaluation of the stored field, and is exact at slice times. Fuzz over fields and query rows, pinning the interval's ends. Verify it fails first with today's `t must not be none` assertion
  - `test_targets_of_a_time_varying_output_are_read_at_a_time_and_a_place`, `oracle_analytic`. It failed first with `t must not be none for time dependent evaluations`. The targets reproduce the manufactured slice values within 1e-6 relative, and equal paired evaluation exactly. Pinned: a single slice with one mode, and 5 or 9 slices with even and odd mode counts.
- [x] 3.2 Implement `targets` for time-varying outputs through paired evaluation. Verify 3.1 passes
  - `targets` routes a time-varying `u` through `evaluate_time_dependent_at` on `u.domain`. Its docstring states the row layout as the contract.
- [x] 3.3 Cover `Predictions of a time-varying output are read at the same points`. SpectralSVR fitted on a time-varying output with an interpolating regressor reproduces the targets on its training functions, row for row, within a measured bound. Fuzz over fields and rows. Verify it fails first with today's `ValueError`
  - `test_predictions_of_a_time_varying_output_are_read_at_the_same_points`. It failed first with `ValueError: x must have shape ... modes is (1,)`. The regressor is a float32 linear LSSVR at C=1e6. float64 set off the library's widening warning, and matching the coefficients' precision is what the library advises. Measured over 60 fields: C=1e4 gave 6.56e-3, C=1e6 gave 6.53e-5, and C=1e8 gave 8.46e-5, so C=1e6. Bound `PREDICTION_RTOL = 5e-4`. The row-refusal scenario is covered alongside, by `test_a_time_varying_output_refuses_a_point_without_a_time`, through both `targets` and `predict`.
- [x] 3.4 Implement `forward` for time-varying outputs. Record the slice count at `train`, reshape predictions to `(n, T, *modes)`, and pass the training field's time interval from `FittedSpectralSVR`, captured at `fit`. Refuse a row without time, as in 2.3, and refuse evaluation before a time-varying fit. Verify 3.3 passes
  - `train` records `output_shape`, the coefficient shape with the time axis, and `forward` reshapes by it. `forward` requires an explicit `domain` for a time-varying output, and lets the basis refuse a row without a time, so one message covers both. `FittedSpectralSVR.output_domain` is captured from `dataset.u.domain` at fit. **Amended:** the refusal before a fit applies to every model, not only time-varying ones. `forward` without `train` used to raise an `AttributeError` on `modes`, and now raises `forward needs a trained model`.
- [x] 3.5 `FittingAndPredicting`, a `RuleBasedStateMachine` marked `oracle_invariant`. It fits one model alternately on time-varying and time-invariant outputs of varying slice counts, and predicts after each fit. The invariant: predictions always match the targets of the most recent fit's field, so no recorded shape or flag leaks from an earlier fit. Verify it fails when the slice count is recorded only on the first fit
  - `FittingAndPredicting` refits one model alternately on time-varying and time-invariant outputs of varying shape. Its invariant is that predictions always match the latest field's targets, within the 3.3 bound. With `output_shape` recorded only on the first fit, it fails.
- [x] 3.6 Cover `A time-invariant output scores as before` in-process. For time-invariant fields, `targets` equals `u(x).real` in float64 exactly, and `forward` equals the grid form's evaluation of the predicted coefficients exactly. Fuzz over fields, mode counts and device. The before-and-after comparison is task 5.2
  - `test_a_time_invariant_output_scores_as_before`: `targets` equals `u(x).real` in float64 exactly, and `forward` equals `FourierBasis.evaluate` of `to_complex_coeff` of the predicted coefficients exactly. The before-and-after comparison is 5.2.

## 4. The forced row

- [ ] 4.1 Cover `The end of the interval is scored`: the forced case's query includes `t = T`, and its targets there equal the final slice of the solution. Fuzz over seed, sizes and device. Verify it fails first against the pilot
- [ ] 4.2 Remove the pilot from `burgers_forced_case`. `u` stays time-dependent, the query takes 12 slice times across `[0, T]` inclusive (every fourth slice in quick mode, every eighth in full) by 12 points in space, and the label loses `pilot: t periodic`. Delete `periodic_in_time` and its test, and update the query-grid test to require slice times across the inclusive interval. Verify 4.1 passes and every existing forced-case test passes against the new layout
- [ ] 4.3 Run the forced row in quick mode on both devices and verify finite metrics and `pred_nan_sum == 0`

## 5. Verification

- [ ] 5.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 5.2 Build a base from a detached worktree of this branch's parent with its own environment, and print the signature of the library each side loads. Run the full table on both with the same stated `--device` and seed. The four time-invariant rows must be bit-identical, except the first DeepONet fit of each run, which is compared against its measured spread. Record the forced row's pilot numbers against its new numbers, with seed, sizes and device, as a deliberate move
- [ ] 5.3 Enumerated mutation over every line this change touches. Run `just mutate` for `SpectralSVR/operators` and `just mutate-scripts <parent>` for the script. The new basis and model code are covered by neither, so first generalise `mutate-scripts` into a recipe taking a module path and a base, and verify it reproduces the script session's scope. Read every survivor. Close each real gap with a test and record each equivalent mutant with its reason
- [ ] 5.4 Verify no source file gained an explanatory prose comment, and that prose was removed where this change touched. Make a second structural pass: no throwaway helpers or private methods, and the structure is the one the code would have if written this way from the start
