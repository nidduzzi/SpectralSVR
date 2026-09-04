## 1. The precision field

- [ ] 1.1 Add a complex-precision field to `SpectralConfig` defaulting to `torch.complex64`, with a derived real precision and a `replace`-based `with_precision`; verify the frozen dataclass still round-trips and `tests/basis/test_config.py` passes untouched
- [ ] 1.2 Verify the full suite is unchanged at this point — nothing consumes the field yet, so any failure here is a defect in the field itself

## 2. Basis side

- [ ] 2.1 Resolve `FourierBasis.coeff_dtype` through the instance configuration instead of the class constant, keeping the attribute readable for `model/SpectralSVR.py` and `problems/burgers.py`; verify a basis with no coefficients still answers, since the config is coefficient-independent
- [ ] 2.2 Route `generate_empty` (`FourierBasis.py:342`) and `generate_coeff` (`:360`) through the resolved precision instead of their `torch.complex64` literals; verify the seeded numerical Burgers field (`nu=0.01`, `(16,16)`, `slice(0,1,50)`, seed 1234) is `torch.equal` to a worktree at this branch's parent commit
- [ ] 2.3 Route `FourierBasis.wave_number`'s `torch.float` default from the resolved real precision; verify `tests/basis` passes in full

## 3. Model side and the boundary

- [ ] 3.1 Thread the real precision from the same decision into `MultiRegression` (`model/__base.py:43`), `FNN` (`:35`) and `LSSVR` (`:140`), keeping `torch.float32` as each default so existing call sites are unaffected; verify `tests/model` passes untouched
- [ ] 3.2 Make `model/SpectralSVR.py:273` check rather than truncate: if the interleaved real precision and the regressor's precision disagree, raise naming both; verify the error text names both precisions and that the matching case is unaffected
- [ ] 3.3 Add a test pairing a double-precision basis with a single-precision regressor and asserting the raise; verify it fails loudly rather than silently downcasting as it does today

## 4. Bridge hardening

- [ ] 4.1 Add `.contiguous()` inside `to_complex_coeff` before the complex view; verify a transposed float64 view converts instead of raising `Tensor must have a last dimension with stride 1`, and that contiguous input is bit-identical to before
- [ ] 4.2 Add an optional target complex precision to `to_complex_coeff`, defaulting to inference so every existing caller is unchanged; verify float32 input with an explicit complex128 target produces complex128
- [ ] 4.3 Route the ETDRK4 fallback (`utils/__init__.py:611`) and the GRF sampler default (`scripts/run_benchmark.py:104`) from the seam rather than their literals; verify `tests/utils/test_solvers.py` passes and `python scripts/run_benchmark.py --quick --operators spectralsvr` still writes finite metrics

## 5. Double precision end to end

- [ ] 5.1 Add `test_double_precision_survives_the_model_boundary` (`oracle_consistency`, `no_fuzz`): construct a double-precision basis, convert coefficients through the bridge, and assert the values reaching the regressor are float64 and exactly equal to the source; verify it fails on the pre-change code, where the boundary truncates
- [ ] 5.2 Add a round-trip test at both precisions over two- and three-axis coefficient grids asserting `torch.equal`; verify it passes
- [ ] 5.3 Document the throughput and memory cost at the point the precision is chosen, per the capability requirement; verify the text names the measured penalty and the kind of work that justifies it

## 6. Verification

- [ ] 6.1 Verify all four benchmark rows are unchanged by regenerating and comparing against the committed table — the default did not move, so any difference is a defect
- [ ] 6.2 Run `just check`; verify clean
- [ ] 6.3 Verify no source file gained an explanatory prose comment; the precision decision must read from the config field's name and the boundary check's error text
