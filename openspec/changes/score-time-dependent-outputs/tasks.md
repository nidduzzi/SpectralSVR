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
  - Re-measured by the review with the same loop and seeds: 9.19e-5. The gap varies from run to run, between 6.53e-5 and 9.19e-5 so far, and stays five to eight times under the 5e-4 bound.
- [x] 3.4 Implement `forward` for time-varying outputs. Record the slice count at `train`, reshape predictions to `(n, T, *modes)`, and pass the training field's time interval from `FittedSpectralSVR`, captured at `fit`. Refuse a row without time, as in 2.3, and refuse evaluation before a time-varying fit. Verify 3.3 passes
  - `train` records `output_shape`, the coefficient shape with the time axis, and `forward` reshapes by it. `forward` requires an explicit `domain` for a time-varying output, and lets the basis refuse a row without a time, so one message covers both. `FittedSpectralSVR.output_domain` is captured from `dataset.u.domain` at fit. **Amended:** the refusal before a fit applies to every model, not only time-varying ones. `forward` without `train` used to raise an `AttributeError` on `modes`, and now raises `forward needs a trained model`.
- [x] 3.5 `FittingAndPredicting`, a `RuleBasedStateMachine` marked `oracle_invariant`. It fits one model alternately on time-varying and time-invariant outputs of varying slice counts, and predicts after each fit. The invariant: predictions always match the targets of the most recent fit's field, so no recorded shape or flag leaks from an earlier fit. Verify it fails when the slice count is recorded only on the first fit
  - `FittingAndPredicting` refits one model alternately on time-varying and time-invariant outputs of varying shape. Its invariant is that predictions always match the latest field's targets, within the 3.3 bound. With `output_shape` recorded only on the first fit, it fails.
- [x] 3.6 Cover `A time-invariant output scores as before` in-process. For time-invariant fields, `targets` equals `u(x).real` in float64 exactly, and `forward` equals the grid form's evaluation of the predicted coefficients exactly. Fuzz over fields, mode counts and device. The before-and-after comparison is task 5.2
  - `test_a_time_invariant_output_scores_as_before`: `targets` equals `u(x).real` in float64 exactly, and `forward` equals `FourierBasis.evaluate` of `to_complex_coeff` of the predicted coefficients exactly. The before-and-after comparison is 5.2.

## 4. The forced row

- [x] 4.1 Cover `The end of the interval is scored`: the forced case's query includes `t = T`, and its targets there equal the final slice of the solution. Fuzz over seed, sizes and device. Verify it fails first against the pilot
  - `test_the_end_of_the_interval_is_scored`. It failed first against the pilot (`assert 0 > 0`, no query at `t = T`). The targets at `t = T` equal the final slice evaluated in space. Measured worst relative gap over 40 cases: 4.51e-7. Bound 5e-6.
- [x] 4.2 Remove the pilot from `burgers_forced_case`. `u` stays time-dependent, the query takes 12 slice times across `[0, T]` inclusive (every fourth slice in quick mode, every eighth in full) by 12 points in space, and the label loses `pilot: t periodic`. Delete `periodic_in_time` and its test, and update the query-grid test to require slice times across the inclusive interval. Verify 4.1 passes and every existing forced-case test passes against the new layout
  - Done in `21c9bb7`. The query takes 12 slice times across `[0, T]` inclusive, and `periodic_in_time` is deleted with its test. The query-grid test requires times on slices across the inclusive interval with the end present. The measured slice-index offset is exactly 0, so that is asserted with `torch.equal`. `50d4f18` added a requirement for interior times.
- [x] 4.3 Run the forced row in quick mode on both devices and verify finite metrics and `pred_nan_sum == 0`
  - Quick runs at seed 0. cpu: SpectralSVR r2 0.803, rrse 0.455; DeepONet r2 0.226. cuda: SpectralSVR r2 0.853, rrse 0.277; DeepONet r2 0.019. `pred_nan_sum` 0 throughout. The cuda run first failed on a device mismatch in paired evaluation, fixed in `9d0796c` with a test that failed first.

## 5. Verification

- [x] 5.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
  - `just check`: ruff, mypy and ty are clean, and 370 passed after `50d4f18`. That commit's message says 369, which is a miscount; the gate printed 370.
- [x] 5.2 Build a base from a detached worktree of this branch's parent with its own environment, and print the signature of the library each side loads. Run the full table on both with the same stated `--device` and seed. The four time-invariant rows must be bit-identical, except the first DeepONet fit of each run, which is compared against its measured spread. Record the forced row's pilot numbers against its new numbers, with seed, sizes and device, as a deliberate move
  - The base came from a worktree of `d525073` with its own `just sync`. Each side printed which library it loaded: the base lacks `evaluate_time_dependent_at`, the new side has it. The full table ran with both operators, `--device cpu` and seed 0, once on the base and twice on the new side. MMS, numerical and GRF, both operators, are byte-identical across all three runs. `antiderivative` SpectralSVR is byte-identical between base and new, and the second new run differs by 1.36e-5 relative, which is run-to-run noise within identical code. `antiderivative` DeepONet is the first DeepONet fit of each run: r2 0.571 on the base, 0.713 and 0.574 on the new side, within the earlier measured spread of 0.159 to 0.797. The forced row moved deliberately, and reproducibly across the two new runs, with seed 0 and full sizes (60, 15, 12, 89 time steps). SpectralSVR: r2 0.9900 to 0.9894, rrse 0.1428 to 0.1387. DeepONet: r2 0.7998 to 0.7423, rrse 0.4217 to 0.5367.
- [x] 5.3 Enumerated mutation over every line this change touches. Run `just mutate` for `SpectralSVR/operators` and `just mutate-scripts <parent>` for the script. The new basis and model code are covered by neither, so first generalise `mutate-scripts` into a recipe taking a module path and a base, and verify it reproduces the script session's scope. Read every survivor. Close each real gap with a test and record each equivalent mutant with its reason
  - `just mutate-changed <config> feat/burgers-forced-benchmark` (`86bda4b`) ran with the basis, model, operators and script configs. The operators config's timeout was raised from 60 s to 600 s, because its baseline measured 56 s. 284 mutants were in scope: 244 killed and 40 survived. Every survivor was re-run by applying its recorded diff against the strengthened tests. The 21 real gaps are all now killed (`50d4f18`): intervals only ever (0, 1), the real-coefficient branch, the default domain, row checks, too few columns, and query times collapsed to {0, 1}. The 19 equivalent survivors still survive, which confirms the classification. They are the lerp weight's divisor of 1 or 0, bool `+` or `^` where both conditions can never hold, `is not` on small ints, and 12 stride and step rewrites that give the same stride.
- [x] 5.4 Verify no source file gained an explanatory prose comment, and that prose was removed where this change touched. Make a second structural pass: no throwaway helpers or private methods, and the structure is the one the code would have if written this way from the start
  - Prose added: contract docstrings only. Four are one-liners, and `forward`'s five-line contract replaced a 15-line template docstring. Removed where touched: the `## Fourier basis` section comment, the TODO, and four prose comments in the grid branch and `forward`, plus the template docstring and the pilot's docstrings. Second structural pass: `fractional_slice_index` has two callers and names the shared index map, `evaluate_time_dependent_at` sits beside the grid form, `FittedSpectralSVR` gains one field, the test builders are shared by two modules, and there are no private helpers.

## 6. Review fixes

A read-only fork reviewed the change adversarially. Each finding below was
reproduced before it was accepted, and the decisions are recorded in design.md.

- [ ] 6.1 Refuse invalid times, each property fuzzed with a test that fails first. Clamp the fractional slice index to `[0, T - 1]` inside `fractional_slice_index`, so a float32 time at `t1` stops reading past the last slice, in the paired and grid forms alike. Refuse non-finite times, a domain without a time interval, and a query of zero points, each with a `ValueError` naming the `(t, x)` layout
- [ ] 6.2 Derive `modes` from `output_shape` and `time_dependent` instead of storing both. Verify the refit state machine and every model test pass unchanged
- [ ] 6.3 Move `fractional_slice_index` beside `Basis.interpolate_time_tensor`, because the inclusive-slice time convention belongs to every basis. Verify the grid and paired tests pass unchanged
- [ ] 6.4 Remove the three prose comments in the regions of `SpectralSVR.py` this change touched: the class-level block, the `features` comment and the one in `train`
- [ ] 6.5 Anchor the prediction test on the manufactured slice values at slice times, as `oracle_analytic`, instead of the library's targets. Measure its gap before setting the bound
- [ ] 6.6 Run `just check`. Run `just mutate-changed` over the lines section 6 touched, read every survivor, and close real gaps. Audit prose, and make a second structural pass

