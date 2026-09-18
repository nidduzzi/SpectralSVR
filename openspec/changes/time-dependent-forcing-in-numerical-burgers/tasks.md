## 0. Dependency gate

- [x] 0.1 Confirm `align-spectral-residual-with-solver` is merged beneath this branch: `git log --oneline` shows its implementation commit and `Burgers.spectral_residual` routes through the shared nonlinear-term helper (real part taken, de-aliasing applied); verify before touching group 4 — without it the round-trip error is flat at 2.59e-3 to 6.89e-3 at every step count and no rate exists to measure. The earlier ETDRK4 stage-time gate is satisfied: that change is archived and in `main`.

## 1. Forcing normalisation in `_solve_spectral`

- [x] 1.1 Introduce a `forcing_at(t) -> (n, ns)` closure inside `_solve_spectral`; for a `Number` forcing it returns the precomputed `f_hat` tensor computed exactly as today; verify the seeded byte-identity check (`nu=0.01`, `(16,16)`, `slice(0,1,50)`, seed 1234) is `torch.equal` against a worktree at this branch's parent commit on both `u` and `f`
- [x] 1.2 Accept a callable forcing: `forcing_at` calls it with the stage time and asserts the result shape is `(n, ns)`; verify a callable returning a constant spectrum reproduces the `Number` path's output to `1e-12`
- [x] 1.3 Accept a fully spectral Basis forcing (`time_dependent=False`): `forcing_at` evaluates the field on `(t, spatial grid of ns points)` via the Basis evaluation path and applies `basis.transform` spatially; verify a forcing field built from a constant reproduces the `Number` path to `1e-10`, and one built at a different spatial resolution than `ns` still returns `(n, ns)`
- [x] 1.4 Replace the constant `+ f_hat` in `nonlinear(ti, v)` with `+ forcing_at(ti)`; verify `tests/problems` still passes in full

## 2. Rejection paths

- [x] 2.1 Raise `ValueError` when a Basis forcing has `time_dependent=True`, naming the two accepted forms in the message. The raise in place today is a `NotImplementedError`, which is the wrong type: a per-time-slice field is invalid input for this branch rather than a form awaiting implementation, and task 2.2's sibling rejection already raises `ValueError`. Changing it before the test is written, not after, so the test does not pin the inconsistency. Verify a test asserts both the type and the message
- [x] 2.2 Raise when a spectral forcing field's `domain[0]` does not equal `(time_domain.start, time_domain.stop)`; verify a test asserts the raise
- [x] 2.3 Remove the `NotImplementedError("numerical Burgers supports a constant forcing f")` branch; verify `grep` finds no remaining reference and `ty check SpectralSVR scripts` is clean with the widened `f` annotation on `generate`
- [x] 2.4 Raise `ValueError` when a spectral forcing field's `domain[1]` does not equal the integration spatial interval. Found by review, not planned: `_forcing_at` validates `domain[0]` only, so a field labelled on `(0.0, 2.0)` integrated against a run on `(0.0, 1.0)` is accepted and returns plausible numbers — measured `f_gen` absolute maximum `38.586` with no raise. The nodes are built from the *integration* interval, so a field on another interval is sampled at points that mean something else on its own periodicity, and the run is silently wrong rather than refused. This is the same failure the time check exists to prevent; only half of it landed. State the scenario `Mismatched space domain raises` in this change's `burgers-time-dependent-forcing` delta **in the same commit as the test**, so the coverage guard never sees it uncovered. Verify a test asserts the raise, and that the same field on a matching domain still integrates
- [x] 2.5 Close the forcing union at the `_forcing_at` boundary so the type refuses what the function refuses. `ForcingInput = ParamInput | ForcingCallable` admits `torch.Tensor` and `Literal["random"]`, both refused at run time — `"random"` never arrives because the manufactured branch intercepts it, and a bare tensor raises. `generate`'s `f` must keep accepting `"random"` as the manufactured trigger, so the narrower alias belongs on `_solve_spectral` and `_forcing_at`, not on `generate`. Follows `Band = RealBand | ExactModeCount` in `utils`, where enumerating the union is exhaustive rather than a sample. Verify `ty check SpectralSVR scripts` is clean and the bare-tensor rejection carries a test

## 3. Returned forcing field

- [x] 3.1 For callable and spectral forcings, build the returned `f_gen` by evaluating `forcing_at` on the output time grid, giving a `(n, nt, ns)` time-dependent field like the solution; verify its values at the grid times equal the input forcing evaluated there to `5e-6`, recorded as `FORCING_ROUND_TRIP_TOLERANCE`. The `1e-10` this task first named is unreachable rather than merely tight: coefficients are stored at complex64, and the transform round trip measures `2.384e-07` to `3.576e-07` across `ns ∈ {16, 24, 32}` and `nt ∈ {20, 40, 60}`, so the bound sits one order above the measurement. The same tests cover the spatial resampling a forcing field at a different mode count receives, which no task named and nothing checked
- [x] 3.2 Keep the `Number` case's `f_gen` construction byte-for-byte; verify task 1.1's byte-identity check still holds after group 3

## 4. MMS round-trip anchor

- [x] 4.1 Add `test_numerical_burgers_reproduces_manufactured_solution` (`oracle_analytic`, `no_fuzz`) in `tests/problems/test_problems.py`: generate a manufactured pair at `modes=(m, m)` with `m = 24`, resample `u(0)` and `f` to `ns = 2m` spatial modes, integrate with `nt = 97`, compare numerical `u(T)` to manufactured `u(T)`; record the measured relative error in the test and assert one order of magnitude above it; verify it passes. Measured at seed 1234: `3.342465e-06`, so the bound is `3e-5`. `nt = 97` is the top of the range 4.2 resolves, not an arbitrary choice — beyond it the error is round-off rather than time-stepping error. `ns = 2m` is headroom, not the condition: the de-aliasing condition is `ns ≥ 6K` for the manufactured cutoff `K = m // 6`, which is at most `m`. The independence the margin was meant to buy has since been measured directly — orders across `ns ∈ {m, 2m, 3m}` at `m ∈ {12, 18, 24, 36}` agree to within 0.1 and errors to about 5% — so the margin is kept because it costs nothing, not because the rate needs it
- [x] 4.2 Measure the convergence order **before** writing any assertion: run the round-trip at `nt ∈ {13, 25, 49, 97}` and record every measured order in the table below, the way `explicit-band-type` recorded `CLOSED_FORM_TOLERANCE`. Measured at `m = 24`, `ns = 2m`, `nu = 0.01`, `n = 2`, seed 1234, on `[0, 1]`:

  | `nt` | `h = 1/(nt-1)` | relative error | order |
  | --- | --- | --- | --- |
  | 13 | 1/12 | 1.401947e-02 | — |
  | 25 | 1/24 | 9.606420e-04 | 3.867 |
  | 49 | 1/48 | 5.935034e-05 | 4.017 |
  | 97 | 1/96 | 3.342465e-06 | 4.150 |
  | 193 | 1/192 | 1.594647e-06 | 1.068 |
  | 385 | 1/384 | 1.416505e-06 | 0.171 |
  | 769 | 1/768 | 2.284473e-06 | -0.690 |

  The step counts are chosen so the step `h = 1/(nt - 1)` halves **exactly**. An earlier revision of this task swept `nt ∈ {13, 25, 50, 100}`, whose steps are `1/12, 1/24, 1/49, 1/99` — ratios of 2.000, 2.042 and 2.020, not 2 — and read the order as `log2` of the error ratio, which assumes exact halving. That inflated two of the three orders, reporting 4.160 and 4.022 where the ratio-corrected values are 4.040 and 3.964. The sweep now halves the step, so `log2` and the ratio-corrected formula agree, and the order helper takes the ratio from the step sizes rather than assuming it. The scenario says "at halving step sizes"; the old sweep did not meet its own wording.

  The last three rows record where the sweep must stop and why; they are not evidence of a rate. They are a round-off floor near `1.5e-6` that rises again as steps accumulate, which is how it is told from a fixed spatial truncation. This task was first written around `nt ∈ {50, 100, 200, 400}`, reasoned from the scheme's nominal order rather than measured; on the halving family the equivalent rows are `nt = 193` and `385`, both floor. The `> 3.5` in the original plan was likewise a prediction. The fourth order it predicted is real, but only over `13 ≤ nt ≤ 97`
- [x] 4.3 Add `test_numerical_burgers_round_trip_converges_at_the_integrator_order` (`oracle_invariant`, `no_fuzz`) asserting that **every** order measured across 4.2's resolved range sits at or above a bound fixed beneath the minimum measured there — that minimum is 3.867, so the bound is 3.8 — the measured orders being 3.867, 4.017 and 4.150. Record all orders in the test. Note the direction: the *bound* sits beneath the measurement, the *measured orders* sit above the bound. Not "the last two halvings", which this task originally said: under a sweep that runs into the floor the last pair is the saturated one, and the assertion would have read against 0.045. A measured order materially below 4 is a finding to chase — de-aliasing resolution, stage times — not a bound to lower. Verify by running it and by task 4.4's control, which supplies the contrast this assertion needs
- [x] 4.4 Add `test_numerical_burgers_forcing_frozen_at_step_start_loses_the_order` (`oracle_invariant`, `no_fuzz`): repeat 4.2's step-count sweep with `forcing_at` wrapped so all four stages of a step read the step's start time, and assert the measured order collapses relative to 4.3's. Record both orders. A separate test rather than a parametrised leg of 4.3: 4.3 claims a rate against the integrator's design order, this claims a differential between two integrations, and one marker should not stand for two kinds of claim. This is the only test of the scenario `Forcing is evaluated at stage times, not step starts`, which nothing covers today
- [x] 4.5 Verify `tests/test_oracle_coverage.py` passes and the three anchors carry `oracle_analytic` / `oracle_invariant`, not `oracle_consistency`

## 5. Documentation

- [x] 5.1 Update `Burgers.generate`'s docstring for `f`: the three accepted forms, the rejected per-slice form and why, and that the forcing is evaluated at the integrator's stage times; verify the text matches the rejection messages in group 2
- [x] 5.2 Update `_solve_spectral`'s docstring to describe `forcing_at` and the resolution condition for de-alias-free manufactured forcing, derived rather than asserted: the manufactured cutoff is `K = m // 6`, so the product reaches `|k| ≤ 2K` while the solver's 2/3 mask at `ns` keeps `|k| ≤ ns/3`, giving `ns ≥ 6K`, which is at most `m`. Verify the condition stated matches design.md's derivation and that task 4.1's `ns = 2m` reads as headroom above it, not as the condition

## 6. Scenario coverage

This change predates both `tests/test_scenario_coverage.py` and the
`test-oracle-convention` requirement that scenarios ranging over inputs are
fuzzed and sequences are modelled statefully. Its 12 scenarios currently sit in
that guard's `EXEMPT` list as `IN_FLIGHT` debt. Each task below is done **as its
group completes**, not in one pass at the end, so the guard is green throughout
and the debt visibly shrinks.

- [x] 6.1 As group 2 completes, mark `Sampled-time field raises` and `Mismatched time domain raises` with `@pytest.mark.scenario` and remove their `IN_FLIGHT` entries; verify the stale-exemption test forces the removal rather than tolerating it. Task 2.4's `Mismatched space domain raises` is stated and covered in one commit, so it never enters `EXEMPT` and has nothing to remove here
- [x] 6.2 As group 3 completes, mark `Callable forcing is integrated`, `Fully spectral forcing field is integrated` and `Constant forcing byte-identity`, and remove their entries. `Constant forcing byte-identity` as first written compared this change against its parent commit, which no test can do; its scenario is restated as the in-suite guarantee it stands for -- the `Number` path returns exactly the constant spectrum, `torch.equal` with no tolerance -- and the cross-commit check stays a task-level verification in 1.1 and 7.3
- [x] 6.3 As group 4 completes, mark `Round-trip matches the manufactured solution` (4.1), `Round-trip converges at the integrator's order` (4.3), `Forcing is evaluated at stage times, not step starts` (4.4 — the control run is what discharges it; no other test reaches that scenario), `Exact solution reached within time-stepping error` (4.1) and `Rate, not just size, is asserted` (4.3), and remove their `IN_FLIGHT` entries
- [x] 6.4 Fuzz the scenarios that range over inputs rather than pinning one case: the forcing-form acceptance over mode counts, sample counts and step counts, and the rejection paths over the shapes that should be refused. Verify each property fails under a mutation of the behaviour it describes.

  The acceptance properties fuzz `space_modes`, `nt`, the sample count `n` and, for a field forcing, the field's own spatial mode count independently of the solver's. The rejection properties fuzz the offending domain offset alongside the mode and step counts, and the space-domain property also integrates the same field on a matching domain so it cannot pass by refusing everything. The anchors in group 4 stay `no_fuzz`: each is a measured rate at stated step counts, which is a fixed-design anchor rather than a property over a range.

  Nine mutations run against the properties, each restored afterwards. All nine killed, no survivors:

  | mutation | property | result |
  | --- | --- | --- |
  | space-domain check removed | `mismatched_space_domain` | killed |
  | time-domain check removed | `mismatched_time_domain` | killed |
  | per-slice rejection removed | `per_time_slice_field` | killed |
  | bare tensor accepted | `rejects_a_bare_tensor` | killed |
  | manufactured forcing let through | `rejects_a_manufactured_forcing` | killed |
  | callable branch returns zero | `callable_forcing_is_integrated` | killed |
  | field branch pinned at `t = 0` | `spectral_forcing_field_is_integrated` | killed |
  | constant spectrum doubled | `constant_forcing_is_the_exact` | killed |
  | stages read the step start | `frozen_at_step_start` | killed |
- [x] 6.5 Decide and record whether the round-trip warrants a `RuleBasedStateMachine`. **No.** The rule applies where a requirement constrains what stays true *across* a sequence rather than within one call. The round-trip sweep looks like a sequence and is not one: each `nt` is an independent integration from the same initial slice, nothing carries between them, and any order is admissible as a starting point. What is constrained is the *rate* relating their errors, which is a single derived quantity over a fixed design, not an invariant a sequence of operations must preserve. A machine here would generate orderings that cannot differ and states that do not exist. The sequences this change does produce — the integrator's own stage calls within a step — are internal to `etdrk4_solver`, which this change does not modify and whose own capability owns them
- [x] 6.6 Verify no `IN_FLIGHT` entry for this change remains, and that the two scenarios belonging to the forced-dataset change moved with it rather than being dropped

## 7. Verification

- [x] 7.1 Run `just check`; verify clean
- [x] 7.2 Verify no source file gained an explanatory prose comment; the three forcing forms must read from `forcing_at`'s structure and the docstrings
- [x] 7.3 Re-run task 1.1's byte-identity check against this branch's *current* parent, which changes when the branch is rebased; verify `torch.equal` on both coefficients. Run against parent `9bb25ea` (`feat/manufactured-band-derivation`) in a detached worktree with its own `uv sync` environment, printing each side's loaded `SpectralSVR.__file__` to confirm which library produced the numbers. At `n=4`, `modes=(16, 16)`, `nu=0.01`, `time_domain=slice(0, 1, 50)`, seed 1234, `f=0`: `torch.equal` **True** on both `u` and `f`, `max|delta| = 0.000e+00`. The changes beneath do not touch `generate_coeff`, so unchanged output is the expectation, but re-measure rather than assume. Re-run task 4.2's order sweep in the same pass and confirm the recorded table still holds, or re-record it: the order depends on the manufactured band, which `manufactured-band-derivation` sets beneath this branch, and a table measured against a parent nobody re-checked is a bound sitting on an assumption
