# Anchor Tests to Independent Oracles

## Why

A test-suite audit found that while the Fourier transform legs are now independently anchored (vs `torch.fft`), several numerical APIs are covered only by self-referential (consistency) or smoke tests — the same weakness that let the Fourier transform rescale bug and the Burgers residual bugs survive. A bug that affects both sides of a round-trip, or that only changes values (not shapes), is currently invisible. The benchmark run on 2026-07-16 also surfaced an actively suspicious `rrse` metric (blew up to ~124k while r2 = 0.987 on the same predictions) whose formula has never been tested against a hand-computed value.

## What Changes

- Add independent/analytic/invariant test anchors for every currently unanchored numerical API:
  - `SpectralSVR.inverse` / `inverse_coeff` — recover a known input from an analytically constructed linear operator (currently shape/no-nan only).
  - Time-axis `grad`/`integral` on time-dependent fields — anchor against closed-form time derivatives (currently shape-only).
  - `resize_modes`, `zero_coeff`, `to_time_dependent`/`to_time_independent` — verify actual coefficient values after truncate/pad/interpolate against hand-computed spectra (currently shape/count only).
  - Non-FFT matmul transform leg and per-axis sampling dispatch — anchor directly against closed-form spectra, not just against the FFT leg.
  - `utils.get_metrics` formulas (`r2`, `rse`, `rrse`, `smape`, ...) — hand-computed values on tiny fixed arrays; investigate and resolve the observed `rrse` blow-up.
  - `LSSVR.fit`/`predict` — closed-form kernel-ridge/normal-equation oracle on a tiny problem (and/or sklearn `KernelRidge` as an independent implementation).
  - `SpectralSVR.forward`/`train` accuracy — analytic linear-operator fixture (antiderivative of a single sine has an exact coefficient answer), replacing reliance on tuned end-to-end SMAPE thresholds alone.
- Introduce a pytest oracle-marker convention: every test declares its oracle class (`independent`, `analytic`, `invariant`, `consistency`, `smoke`) via markers registered in `tests/conftest.py`.
- **BREAKING (test suite convention)**: retire the `mms`/`no_mms` markers — `oracle_analytic` subsumes the MMS declaration; the fuzz policy stays unchanged.
- Enforce the convention: unmarked tests fail collection, and a guard checks that each covered module has at least one non-consistency/non-smoke anchor.
- Grow the shared anchors library (`SpectralSVR/problems/exact.py` + `tests/problems/_exact.py` pattern) so anchors are reusable across test modules.

## Capabilities

### New Capabilities
- `test-oracle-convention`: pytest marker taxonomy for test oracle classes, registration, collection-time enforcement, and per-module anchor coverage guard.
- `numerical-anchor-tests`: independent/analytic/invariant anchor tests for the unanchored numerical APIs listed above, including the metric-formula verification and the rrse investigation.

### Modified Capabilities

<!-- none: no runtime spec-level behavior changes; if the rrse investigation reveals a genuine formula bug in utils.get_metrics, the fix is a bug fix restoring intended behavior, captured in tasks -->

## Impact

- `tests/` — all test modules gain oracle markers; new anchor test files/cases in `tests/basis/`, `tests/model/`, `tests/utils/`, `tests/operators/`.
- `tests/conftest.py` — marker registration + enforcement hooks (existing `pytest_configure`/`pytest_runtest_setup` extended).
- `SpectralSVR/problems/exact.py` and/or `tests/` helper modules — new closed-form anchor helpers.
- `SpectralSVR/utils/__init__.py` (`get_metrics`) — possible bug fix if the rrse formula is wrong.
- No public API changes intended; possible optional dev-dependency (sklearn) if chosen as the LSSVR oracle.
