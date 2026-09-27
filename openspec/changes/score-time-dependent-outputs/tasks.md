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

- [ ] 2.1 Invariant, `oracle_invariant`: a time-varying field evaluated at paired rows `(t_i, x_j)` equals the grid form's `out[:, i, j]`, for every `i` and `j`. Fuzz over the number of functions, slice count, mode count (even and odd) and device. Pin `t = t0`, `t = t1`, `x = 0`, `x = L`, a single time slice, and the smallest mode counts. Measure the gap before setting its bound. Verify it fails first
- [ ] 2.2 Analytic anchor, `oracle_analytic`: a field manufactured from known values on the spatial grid at every slice gives those values exactly at slice times. Midway between two slices it gives their mean, by the stated linear convention. Fuzz as 2.1, with the same pins. Verify it fails first
- [ ] 2.3 A row carrying only spatial coordinates is refused for a time-varying field, with a message naming the `(t, x)` layout. Fuzz over spatial dimension and row count. Verify it fails first
- [ ] 2.4 Implement paired evaluation beside the grid form, per the design. Verify 2.1 to 2.3 pass, and that the grid form's existing tests pass unchanged

## 3. Targets and forward

- [ ] 3.1 Cover `Targets of a time-varying output are read at a time and a place`: `targets` on a time-varying output equals paired evaluation of the stored field, and is exact at slice times. Fuzz over fields and query rows, pinning the interval's ends. Verify it fails first with today's `t must not be none` assertion
- [ ] 3.2 Implement `targets` for time-varying outputs through paired evaluation. Verify 3.1 passes
- [ ] 3.3 Cover `Predictions of a time-varying output are read at the same points`. SpectralSVR fitted on a time-varying output with an interpolating regressor reproduces the targets on its training functions, row for row, within a measured bound. Fuzz over fields and rows. Verify it fails first with today's `ValueError`
- [ ] 3.4 Implement `forward` for time-varying outputs. Record the slice count at `train`, reshape predictions to `(n, T, *modes)`, and pass the training field's time interval from `FittedSpectralSVR`, captured at `fit`. Refuse a row without time, as in 2.3, and refuse evaluation before a time-varying fit. Verify 3.3 passes
- [ ] 3.5 `FittingAndPredicting`, a `RuleBasedStateMachine` marked `oracle_invariant`. It fits one model alternately on time-varying and time-invariant outputs of varying slice counts, and predicts after each fit. The invariant: predictions always match the targets of the most recent fit's field, so no recorded shape or flag leaks from an earlier fit. Verify it fails when the slice count is recorded only on the first fit
- [ ] 3.6 Cover `A time-invariant output scores as before` in-process. For time-invariant fields, `targets` equals `u(x).real` in float64 exactly, and `forward` equals the grid form's evaluation of the predicted coefficients exactly. Fuzz over fields, mode counts and device. The before-and-after comparison is task 5.2

## 4. The forced row

- [ ] 4.1 Cover `The end of the interval is scored`: the forced case's query includes `t = T`, and its targets there equal the final slice of the solution. Fuzz over seed, sizes and device. Verify it fails first against the pilot
- [ ] 4.2 Remove the pilot from `burgers_forced_case`. `u` stays time-dependent, the query takes 12 slice times across `[0, T]` inclusive (every fourth slice in quick mode, every eighth in full) by 12 points in space, and the label loses `pilot: t periodic`. Delete `periodic_in_time` and its test, and update the query-grid test to require slice times across the inclusive interval. Verify 4.1 passes and every existing forced-case test passes against the new layout
- [ ] 4.3 Run the forced row in quick mode on both devices and verify finite metrics and `pred_nan_sum == 0`

## 5. Verification

- [ ] 5.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 5.2 Build a base from a detached worktree of this branch's parent with its own environment, and print the signature of the library each side loads. Run the full table on both with the same stated `--device` and seed. The four time-invariant rows must be bit-identical, except the first DeepONet fit of each run, which is compared against its measured spread. Record the forced row's pilot numbers against its new numbers, with seed, sizes and device, as a deliberate move
- [ ] 5.3 Enumerated mutation over every line this change touches. Run `just mutate` for `SpectralSVR/operators` and `just mutate-scripts <parent>` for the script. The new basis and model code are covered by neither, so first generalise `mutate-scripts` into a recipe taking a module path and a base, and verify it reproduces the script session's scope. Read every survivor. Close each real gap with a test and record each equivalent mutant with its reason
- [ ] 5.4 Verify no source file gained an explanatory prose comment, and that prose was removed where this change touched. Make a second structural pass: no throwaway helpers or private methods, and the structure is the one the code would have if written this way from the start
