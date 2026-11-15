# Design: Anchor Tests to Independent Oracles

## Context

An audit (2026-07-16) classified all ~212 tests by oracle type. The Fourier transform legs, wavenumber grid, spatial derivatives, ETDRK4 (linear/MMS/Cole-Hopf), and problem residuals are well anchored. The remaining gaps are covered only by consistency (round-trip / self-generated-data) or smoke (shape/no-crash) tests:

| API | Current coverage | Bug class it would miss |
|---|---|---|
| `SpectralSVR.inverse`/`inverse_coeff` | shape/no-nan | whole feature wrong |
| time-axis `grad`/`integral` (time-dependent) | shape only | wrong stencil/scale, like past Burgers residual bugs |
| `resize_modes`, `zero_coeff`, `to_time_dependent` | shape/count | wrong values after truncate/pad/interp (rescale-bug class) |
| matmul (non-FFT) transform leg, per-axis sampling | compared to FFT leg only | shared convention error |
| `get_metrics` (`r2`,`rse`,`rrse`,`smape`) | key presence, `>=0` | wrong formula — rrse already suspicious (124k vs r2 0.987) |
| `LSSVR.fit/predict` | self-interpolation, tuned tol | wrong solver math |
| `SpectralSVR.forward/train` | tuned end-to-end SMAPE on self-data | systematic pipeline error inside tolerance |

`tests/conftest.py` already enforces a marker policy (fuzz + MMS declarations fail collection when absent) — the oracle convention extends this proven pattern.

## Goals / Non-Goals

**Goals:**
- Every listed gap gets at least one independent/analytic/invariant anchor.
- Every test declares its oracle class; unmarked tests fail at setup, same as the existing fuzz/MMS policy.
- Resolve the rrse anomaly: hand-computed metric values either pass (anomaly is data-driven, document why) or expose a formula bug (fix it).
- Anchor helpers live in one reusable place.

**Non-Goals:**
- Not reworking already-anchored tests (transform legs, residual analytic tests stay as-is; they only gain markers).
- Not adding mutation testing to CI (one-off audit already done manually).
- No runtime behavior changes except a possible `get_metrics` bug fix.
- Not anchoring trivial plumbing (config, plot smoke tests) beyond markers.

## Decisions

### D1. Oracle marker taxonomy — five markers, exactly one required per test
`@pytest.mark.oracle_independent | oracle_analytic | oracle_invariant | oracle_consistency | oracle_smoke`, registered in `pytest_configure`, enforced in `pytest_runtest_setup` next to the fuzz check (raise on zero or multiple markers). Rationale: mirrors the existing enforced-marker pattern; a `@pytest.mark.oracle("analytic")` parametrized marker was considered but plain markers grep better and match the existing style.

The existing `mms`/`no_mms` markers are RETIRED: MMS is exactly "truth manufactured analytically", which `oracle_analytic` subsumes, so keeping both means every test declares the same fact twice. Migration: `mms` → `oracle_analytic` (audit table confirms each case), `no_mms` → whichever oracle class the audit assigned; the `mms`/`no_mms` registration and setup check are removed from conftest. The fuzz policy (`@given`/`no_fuzz`) is orthogonal (input-drawing strategy, not truth source) and stays unchanged.

### D2. Coverage guard as a test, not a plugin
A meta-test (`tests/test_oracle_coverage.py`) walks collected items (via `pytest_collection` hook storing a registry, or a static walk over test modules' markers) and asserts each test module contains ≥1 test marked `oracle_independent|oracle_analytic|oracle_invariant`. Exempt list (explicit, in the meta-test): modules that are pure plumbing (`test_config.py`, `test_sampling.py` if per-axis anchor lands elsewhere — final list decided at implementation). Rationale: a test is visible, reviewable, and needs no pytest plugin machinery.

### D3. Anchor helper placement
Closed-form helpers usable by the library itself (exact solutions) stay in `SpectralSVR/problems/exact.py`; test-only spectra/fixtures (hand-computed Fourier coefficients, tiny metric arrays, normal-equation solves) go to a new `tests/_anchors.py`. Rationale: keeps runtime package free of test-only code while keeping one obvious home per side.

### D4. Concrete oracles per gap
- **inverse**: train SpectralSVR on the antiderivative operator with single-mode sine inputs; `inverse` of the known output must recover the known input coefficients (exact closed form for both sides). Tolerance from optimizer convergence, asserted against the analytic target, not a self-produced one.
- **time-axis grad/integral**: field `u(t,x) = g(t)·sin(2πx)` with polynomial/exponential `g`; time derivative closed-form. Anchor values at interior time points (finite-diff time axis).
- **resize_modes / zero_coeff**: start from a hand-written spectrum (e.g. modes {0:1, 1:2i, 2:−1}); truncation, padding, and zeroing have exact expected coefficient arrays written literally in the test.
- **matmul transform leg**: force the non-FFT path (non-uniform/per-axis sampling or strategy flag) on a single-mode sine; expected spectrum written by hand — not compared to the FFT leg.
- **get_metrics**: 3–4-element fixed arrays; expected `mse/rmse/mae/r2/smape/rse/rrse` computed by hand in comments. Include a case reproducing the rrse blow-up pattern (near-constant target column) to characterize intended behavior; fix formula if it disagrees with the standard definition (rrse = sqrt(SSE/SST) — suspect per-feature normalization by near-zero variance).
- **LSSVR**: linear-kernel LSSVR on a tiny dataset has a closed-form solution via its KKT linear system — solve the same system with `torch.linalg.solve` written inline in the test (independent path), assert predictions match. sklearn dependency avoided (kernel-ridge ≠ LSSVR formulation exactly; inline normal-equation oracle is exact and dependency-free).
- **SpectralSVR.forward**: antiderivative of `sin(2πmx)` is `−cos(2πmx)/(2πm)`; train on a family of single-mode inputs, assert predicted coefficients/values against the closed form (tight tolerance, not tuned-to-pass).

### D5. New anchor tests are fuzzed by default
New anchors follow the suite's existing fuzz policy: Hypothesis `@given` over the closed form's free parameters (mode index `m`, amplitude, phase, viscosity, array sizes, seeds) wherever the oracle stays exact under parametrization — the analytic expectation is computed from the drawn parameters inside the test, so fuzzing widens the anchor instead of loosening it. `@pytest.mark.no_fuzz` is reserved for anchors that are genuinely fixed-point (hand-written literal spectra in resize/zero_coeff, the hand-computed metric tables, the tiny LSSVR KKT system) where a literal expected value IS the point. Rationale: matches the enforced fuzz/MMS policy already in `tests/conftest.py`; a fuzzed analytic anchor is strictly stronger than a fixed-example one.

### D6. Marker rollout for existing tests
All 212 existing tests get their audited classification applied mechanically in one pass (the audit table is the source). Borderline cases resolved conservatively: when in doubt, mark `oracle_consistency` (undersell, never oversell).

## Risks / Trade-offs

- [rrse "fix" changes benchmark numbers] → treat as bug fix with its own task; rerun `bench_results.csv` after; document the definition chosen in the docstring.
- [LSSVR closed-form oracle re-derives the same math as the implementation] → write the KKT system independently from the paper definition in the test, not by importing/refactoring library code; different code path is the point.
- [Marker enforcement breaks the whole suite until rollout completes] → single change lands markers + enforcement together; enforcement raise added last in the implementation order.
- [Coverage guard false-positives on legitimately smoke-only modules] → explicit exempt list, reviewed in PR.
- [Inverse anchor flaky (Adam optimizer)] → fixed seed, generous-but-meaningful tolerance against the analytic target; assert error against truth, not loss decrease.

## Open Questions

- Exact exempt-module list for the coverage guard (decided during implementation when markers are applied).
- Whether rrse blow-up is formula bug or torchmetrics' intended per-output normalization — investigation task answers it.
