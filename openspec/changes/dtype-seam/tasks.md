## 1. The precision field

- [x] 1.1 Add a complex-precision field to `SpectralConfig` defaulting to `torch.complex64`, with a derived real precision and a `replace`-based `with_precision`; verify the frozen dataclass still round-trips and `tests/basis/test_config.py` passes untouched
- [x] 1.2 Verify the full suite is unchanged at this point — nothing consumes the field yet, so any failure here is a defect in the field itself

## 2. Basis side

- [x] 2.1 Resolve `FourierBasis.coeff_dtype` through the instance configuration instead of the class constant, keeping the attribute readable for `model/SpectralSVR.py` and `problems/burgers.py`; verify a basis with no coefficients still answers, since the config is coefficient-independent
- [x] 2.2 Route `generate_empty` (`FourierBasis.py:342`) and `generate_coeff` (`:360`) through the resolved precision instead of their `torch.complex64` literals; verify the seeded numerical Burgers field (`nu=0.01`, `(16,16)`, `slice(0,1,50)`, seed 1234) is `torch.equal` to a worktree at this branch's parent commit
- [x] 2.3 Route `FourierBasis.wave_number`'s `torch.float` default from the resolved real precision; verify `tests/basis` passes in full

## 3. Model side and the boundary

- [x] 3.1 Thread the real precision from the same decision into `MultiRegression` (`model/__base.py:43`), `FNN` (`:35`) and `LSSVR` (`:140`), keeping `torch.float32` as each default so existing call sites are unaffected; verify `tests/model` passes untouched
- [x] 3.2 Make the model boundary check rather than truncate, asymmetrically. **Lossy** — values wider than the regressor, where a cast would discard precision the basis was configured to keep — raises, naming both precisions. **Wasteful** — the regressor wider than the values, where nothing is discarded but the run pays double-precision cost for single-precision data — warns and proceeds. Verify each direction separately, and that the matching case is silent. The guard applies only to values carrying a basis's coefficient precision through the interleaved-real bridge, not to real inputs a caller supplies at whatever precision suits them: those have always been cast and the basis's configuration says nothing about them. Verify that distinction holds by confirming the float64 query grids in `tests/operators/test_operators.py` still pass, since guarding them was what surfaced it
- [x] 3.3 Add tests covering both directions: a double-precision basis with a single-precision regressor raises, and verify it fails loudly rather than silently downcasting as it does today; a single-precision basis with a double-precision regressor warns and still produces a result. Verify the two existing anchors in `tests/model/test_spectralsvr_anchors.py` are the wasteful case in the wild — they pair a `complex64` basis with `LSSVR(dtype=torch.float64)` at lines 51 and 97 because their closed-form comparison needs the precision in the solve — so they must keep passing and should now emit that warning

## 4. The bridge's guarantees, anchored before they are edited

The design requires this section to precede the hardening: `coefficient-encodings`
binds *every* exported conversion to accept any number of mode axes and to
preserve device and precision, and only the polar pair carries tests for it. The
interleaved-real pair satisfies the requirement today with nothing defending it,
and section 5 edits exactly that pair. An anchor written afterwards would be
shaped to match whatever the edit produced.

- [x] 4.1 Add a conversion-registry anchor in `tests/utils/test_utils.py` parametrised over every conversion the package exports — `to_real_coeff`/`to_complex_coeff` and `to_mag_angle`/`from_mag_angle` — asserting for each: multi-axis conversion at two, three and four axes; exact round-trip through its inverse; the real precision corresponding to the complex one at both `complex64` and `complex128`; and the device carried through, skipping cleanly when no accelerator is present. Build it from a list of `(forward, inverse)` pairs rather than one test per function, so a conversion added later is either covered or fails the anchor. Verify the registry covers all four exported conversions by name, and that removing one from the package's exports makes the anchor fail rather than silently shrink
- [x] 4.2 Verify 4.1 passes against the **unmodified** bridge, so it records the guarantee as it already stands rather than as section 5 leaves it; verify it fails if `to_complex_coeff` is made to drop the device or infer the wrong precision, which is the regression sections 5.1 and 5.2 could introduce

## 5. Bridge hardening

- [x] 5.1 Add `.contiguous()` inside `to_complex_coeff` before the complex view; verify a transposed float64 view converts instead of raising `Tensor must have a last dimension with stride 1`, and that contiguous input is bit-identical to before
- [x] 5.2 Add an optional target complex precision to `to_complex_coeff`, defaulting to inference so every existing caller is unchanged; verify float32 input with an explicit complex128 target produces complex128
- [x] 5.3 Verify the section 4 anchor still passes unchanged after both edits; it was written against the bridge as it stood, so it passing now is evidence the hardening was additive rather than a change of behaviour
- [x] 5.4 Route the ETDRK4 fallback (`utils/__init__.py:611`) and the GRF sampler default (`scripts/run_benchmark.py:104`) from the seam rather than their literals; verify `tests/utils/test_solvers.py` passes and `python scripts/run_benchmark.py --quick --operators spectralsvr` still writes finite metrics

## 6. Double precision end to end

- [x] 6.1 Add `test_double_precision_survives_the_model_boundary` (`oracle_consistency`, `no_fuzz`): construct a double-precision basis, convert coefficients through the bridge, and assert the values reaching the regressor are float64 and exactly equal to the source; verify it fails on the pre-change code, where the boundary truncates
- [x] 6.2 Add a round-trip test at both precisions over two- and three-axis coefficient grids asserting `torch.equal`; verify it passes
- [x] 6.3 Document the throughput and memory cost at the point the precision is chosen, per the capability requirement; verify the text names the measured penalty and the kind of work that justifies it

## 7. Verification

- [ ] 7.1 Verify all four benchmark rows are unchanged by regenerating and comparing against the committed table — the default did not move, so any difference is a defect
- [ ] 7.2 Verify the seeded numerical Burgers field is byte-identical to the baseline captured before any edit; record the hash on both sides
- [ ] 7.3 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 7.4 Verify no source file gained an explanatory prose comment; the precision decision must read from the config field's name and the boundary check's error text
