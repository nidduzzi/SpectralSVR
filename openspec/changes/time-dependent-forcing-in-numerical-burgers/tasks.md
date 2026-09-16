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
- [ ] 4.2 Measure the convergence order **before** writing any assertion: run the round-trip at `nt ∈ {50, 100, 200, 400}` and record every measured order in a table in this file, the way `explicit-band-type` recorded `CLOSED_FORM_TOLERANCE`. The `> 3.5` in the original plan was reasoned from the scheme's nominal order, not measured, and no rate has ever been observed here — the dependency gate's own text records the error as flat at 2.59e-3 to 6.89e-3 with no rate at all before `align-spectral-residual-with-solver` landed
- [ ] 4.3 Add `test_numerical_burgers_round_trip_converges_at_the_integrator_order` (`oracle_invariant`, `no_fuzz`) asserting the order over the last two halvings sits beneath the value measured in 4.2, with all orders recorded in the test. A measured order materially below 4 is a finding to chase — de-aliasing resolution, stage times — not a bound to lower. Verify by reading the recorded values that the same test would read ~1 with all stages at `t_n`
- [ ] 4.4 Verify `tests/test_oracle_coverage.py` passes and the two anchors carry `oracle_analytic` / `oracle_invariant`, not `oracle_consistency`

## 5. Documentation

- [ ] 5.1 Update `Burgers.generate`'s docstring for `f`: the three accepted forms, the rejected per-slice form and why, and that the forcing is evaluated at the integrator's stage times; verify the text matches the rejection messages in group 2
- [ ] 5.2 Update `_solve_spectral`'s docstring to describe `forcing_at` and the `ns ≥ 1.5 m` resolution condition for de-alias-free manufactured forcing; verify the condition stated matches what task 4.1 uses

## 6. Scenario coverage

This change predates both `tests/test_scenario_coverage.py` and the
`test-oracle-convention` requirement that scenarios ranging over inputs are
fuzzed and sequences are modelled statefully. Its 12 scenarios currently sit in
that guard's `EXEMPT` list as `IN_FLIGHT` debt. Each task below is done **as its
group completes**, not in one pass at the end, so the guard is green throughout
and the debt visibly shrinks.

- [ ] 6.1 As group 2 completes, mark `Sampled-time field raises` and `Mismatched time domain raises` with `@pytest.mark.scenario` and remove their `IN_FLIGHT` entries; verify the stale-exemption test forces the removal rather than tolerating it
- [ ] 6.2 As group 3 completes, mark `Callable forcing is integrated`, `Fully spectral forcing field is integrated` and `Constant forcing byte-identity`, and remove their entries
- [ ] 6.3 As group 4 completes, mark `Round-trip matches the manufactured solution`, `Round-trip converges at the integrator's order`, `Forcing is evaluated at stage times, not step starts`, `Exact solution reached within time-stepping error` and `Rate, not just size, is asserted`, and remove their entries
- [ ] 6.4 Fuzz the scenarios that range over inputs rather than pinning one case: the forcing-form acceptance over mode counts, sample counts and step counts, and the rejection paths over the shapes that should be refused. Verify each property fails under a mutation of the behaviour it describes
- [ ] 6.5 Decide and record whether the round-trip warrants a `RuleBasedStateMachine`. Its point is a measured rate at specific `nt` values, which is a fixed-design anchor rather than a sequence invariant — if no sequence is being constrained, record that reasoning rather than adding a machine to satisfy the letter of the rule
- [ ] 6.6 Verify no `IN_FLIGHT` entry for this change remains, and that the two scenarios belonging to the forced-dataset change moved with it rather than being dropped

## 7. Verification

- [ ] 7.1 Run `just check`; verify clean
- [ ] 7.2 Verify no source file gained an explanatory prose comment; the three forcing forms must read from `forcing_at`'s structure and the docstrings
- [ ] 7.3 Re-run task 1.1's byte-identity check against this branch's *current* parent, which changes when the branch is rebased onto `align-spectral-residual-with-solver`; verify `torch.equal` on both coefficients. That change does not touch `generate_coeff`, so unchanged output is the expectation, but re-measure rather than assume
