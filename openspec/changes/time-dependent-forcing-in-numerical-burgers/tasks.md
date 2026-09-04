## 0. Dependency gate

- [ ] 0.1 Confirm `align-spectral-residual-with-solver` is merged beneath this branch: `git log --oneline` shows its implementation commit and `Burgers.spectral_residual` routes through the shared nonlinear-term helper (real part taken, de-aliasing applied); verify before touching group 4 — without it the round-trip error is flat at 2.59e-3 to 6.89e-3 at every step count and no rate exists to measure. The earlier ETDRK4 stage-time gate is satisfied: that change is archived and in `main`.

## 1. Forcing normalisation in `_solve_spectral`

- [x] 1.1 Introduce a `forcing_at(t) -> (n, ns)` closure inside `_solve_spectral`; for a `Number` forcing it returns the precomputed `f_hat` tensor computed exactly as today; verify the seeded byte-identity check (`nu=0.01`, `(16,16)`, `slice(0,1,50)`, seed 1234) is `torch.equal` against a worktree at this branch's parent commit on both `u` and `f`
- [x] 1.2 Accept a callable forcing: `forcing_at` calls it with the stage time and asserts the result shape is `(n, ns)`; verify a callable returning a constant spectrum reproduces the `Number` path's output to `1e-12`
- [x] 1.3 Accept a fully spectral Basis forcing (`time_dependent=False`): `forcing_at` evaluates the field on `(t, spatial grid of ns points)` via the Basis evaluation path and applies `basis.transform` spatially; verify a forcing field built from a constant reproduces the `Number` path to `1e-10`, and one built at a different spatial resolution than `ns` still returns `(n, ns)`
- [x] 1.4 Replace the constant `+ f_hat` in `nonlinear(ti, v)` with `+ forcing_at(ti)`; verify `tests/problems` still passes in full

## 2. Rejection paths

- [ ] 2.1 Raise when a Basis forcing has `time_dependent=True`, naming the two accepted forms in the message; verify a test asserts the raise
- [ ] 2.2 Raise when a spectral forcing field's `domain[0]` does not equal `(time_domain.start, time_domain.stop)`; verify a test asserts the raise
- [x] 2.3 Remove the `NotImplementedError("numerical Burgers supports a constant forcing f")` branch; verify `grep` finds no remaining reference and `ty check SpectralSVR scripts` is clean with the widened `f` annotation on `generate`

## 3. Returned forcing field

- [ ] 3.1 For callable and spectral forcings, build the returned `f_gen` by evaluating `forcing_at` on the output time grid, giving a `(n, nt, ns)` time-dependent field like the solution; verify its values at the grid times equal the input forcing evaluated there to `1e-10`
- [x] 3.2 Keep the `Number` case's `f_gen` construction byte-for-byte; verify task 1.1's byte-identity check still holds after group 3

## 4. MMS round-trip anchor

- [ ] 4.1 Add `test_numerical_burgers_reproduces_manufactured_solution` (`oracle_analytic`, `no_fuzz`) in `tests/problems/test_problems.py`: generate a manufactured pair at `modes=(m, m)`, resample `u(0)` and `f` to `ns = 2m` spatial modes, integrate with `nt = 400`, compare numerical `u(T)` to manufactured `u(T)`; record the measured relative error in the test and assert one order of magnitude above it; verify it passes
- [ ] 4.2 Add `test_numerical_burgers_round_trip_converges_at_fourth_order` (`oracle_invariant`, `no_fuzz`): same round-trip at `nt ∈ {50, 100, 200, 400}`, assert measured order over the last two halvings `> 3.5`, record all orders in the test; verify it passes on the corrected solver and confirm by reading the recorded values that it would read ~1 with all stages at `t_n`
- [ ] 4.3 Verify `tests/test_oracle_coverage.py` passes and the two anchors carry `oracle_analytic` / `oracle_invariant`, not `oracle_consistency`

## 5. Forced-Burgers benchmark dataset

- [ ] 5.1 In `scripts/run_benchmark.py`, add `burgers_forced_case`: `u0 = 0`, forcing from `FourierBasis.generate(n, (mt, ms), random_func=grf_coeff(...), domain=((0, 1), (0, 1)))` with recorded length scales, solution kept `time_dependent=True`, dataset via `OperatorDataset.from_fields(f, u, n_sensors)`; verify it builds in `--quick` mode
- [ ] 5.2 Register it as `DatasetSpec("project Burgers forced (f->u, GRF forcing, from rest)", burgers_forced_case)` under key `burgers-forced`; verify `--help` lists it and `--datasets burgers-forced --operators spectralsvr --quick` writes a row with finite metrics to `results/bench_results.csv`
- [ ] 5.3 Verify `OperatorDataset.targets(query)` on the forced dataset agrees with direct evaluation of the stored coefficients at output-grid `(t, x)` points to `1e-6`; if time interpolation error exceeds that, restrict the dataset's query grid to output-grid times and record why in the `DatasetSpec` docstring

## 6. Documentation

- [ ] 6.1 Update `Burgers.generate`'s docstring for `f`: the three accepted forms, the rejected per-slice form and why, and that the forcing is evaluated at the integrator's stage times; verify the text matches the rejection messages in group 2
- [ ] 6.2 Update `_solve_spectral`'s docstring to describe `forcing_at` and the `ns ≥ 1.5 m` resolution condition for de-alias-free manufactured forcing; verify the condition stated matches what task 4.1 uses

## 7. Verification

- [ ] 7.1 Run `just check`; verify clean
- [ ] 7.2 Verify no source file gained an explanatory prose comment; the three forcing forms must read from `forcing_at`'s structure and the docstrings
- [ ] 7.3 Re-run task 1.1's byte-identity check against this branch's *current* parent, which changes when the branch is rebased onto `align-spectral-residual-with-solver`; verify `torch.equal` on both coefficients. That change does not touch `generate_coeff`, so unchanged output is the expectation, but re-measure rather than assume
