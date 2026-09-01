# Tasks: Anchor Tests to Independent Oracles

> Implementation note: use codegraph (`codegraph_explore`) for locating symbols/callers and the ty LSP for rename/diagnostics — the marker migration (1.2, 1.3) and any `get_metrics` fix (2.3) are mechanical sweeps where LSP rename + codegraph blast-radius checks replace grep/read loops.

## 1. Oracle marker convention

- [x] 1.1 Register the five `oracle_*` markers in `tests/conftest.py` (`pytest_configure`)
- [x] 1.2 Apply oracle markers to all existing tests per the 2026-07 audit table (borderline → `oracle_consistency`); migrate `mms` → `oracle_analytic` and drop all `mms`/`no_mms` markers in the same pass
- [x] 1.3 Remove `mms`/`no_mms` registration and setup enforcement from conftest; add setup-time enforcement of exactly one `oracle_*` marker per test (raise on zero or multiple)
- [x] 1.4 Add `tests/test_oracle_coverage.py` meta-test: every non-exempt module has ≥1 `oracle_independent|analytic|invariant` test; write the explicit exempt list
- [x] 1.5 Full suite green with enforcement on

## 2. Metric formulas + rrse investigation

- [x] 2.1 Hand-computed anchor test for `get_metrics` on small fixed arrays (all metrics, `no_fuzz`, values derived in comments)
- [x] 2.2 Reproduce the rrse blow-up with a near-constant target column; compare implementation to standard definition sqrt(SSE/SST)
- [x] 2.3 If formula wrong: fix `utils.get_metrics`, update docstring with the definition chosen
- [x] 2.4 If fixed: regenerate `bench_results.csv` with the corrected metric

## 3. Anchor helpers

- [x] 3.1 Create `tests/_anchors.py`: hand-written spectra, separable u(t,x)=g(t)·sin(2πx) family with closed-form time derivatives, single-mode antiderivative closed forms
- [x] 3.2 Extend `SpectralSVR/problems/exact.py` only if a helper is reusable by the library; otherwise keep test-side

## 4. Basis-layer anchors (fuzzed via Hypothesis where the closed form parametrizes)

- [x] 4.1 Time-axis `grad`/`integral` anchor: interior values vs closed-form dg/dt·sin(2πx), `@given` over g's parameters and mode
- [x] 4.2 `resize_modes` value anchor: literal expected spectra for truncate and pad (`no_fuzz`)
- [x] 4.3 `zero_coeff` value anchor: literal expected masked spectrum (`no_fuzz`)
- [x] 4.4 `to_time_dependent`/`to_time_independent` value anchor vs closed-form field values
- [x] 4.5 Non-FFT matmul transform leg anchor: forced non-FFT path on `@given`-drawn single-mode sinusoid vs closed-form spectrum (also covers per-axis sampling dispatch)

## 5. Model-layer anchors

- [x] 5.1 LSSVR KKT oracle test: inline `torch.linalg.solve` of the paper-definition KKT system on a tiny problem (`no_fuzz`); assert predictions match
- [x] 5.2 `SpectralSVR.forward` analytic anchor: single-mode sine antiderivative family, `@given` over mode/amplitude, fixed tolerance vs closed form
- [x] 5.3 `SpectralSVR.inverse`/`inverse_coeff` anchor: recover analytically known input (fixed seed), error asserted against analytic target

## 6. Wrap-up

- [x] 6.1 Run full suite + ruff; confirm coverage guard passes and no `mms` references remain (grep)
- [x] 6.2 Update CLAUDE.md / testing-policy docstring in conftest to describe the oracle convention
