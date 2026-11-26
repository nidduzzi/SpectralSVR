## SpectralSVR/model/SpectralSVR.py (22 survivors)

No `from __future__ import annotations` here. `test()` is only asserted on metric keys (and `mse >= 0`), never on metric values. No test calls `inverse_coeff` with `generator=None` and `epochs=0`, or checks the returned tensor's `requires_grad`. No non-time-dependent `forward` is given an explicit `domain`.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 95 | `len(x.shape) == 1`->`<= 1` (1) | dead | differs only for 0-d `x`, unsupported (the next line indexes `x.shape[1]`) |
| 96 | `unsqueeze(-1)`->`unsqueeze(+1)` / `unsqueeze(1)` (2) | equivalent | only reached for 1-D `x`, where dim 1 and dim -1 coincide |
| 97 | `!=`->`is not` (1) | equivalent | small ints are identical below 257; `x.shape[1]` equal to `len(modes)` is a handful of axes |
| 122 | `domain is not None`->`is None` / `not ... is not` (2) | gap | `forward` with an explicit `domain` differing from `config.domain` is never run on a non-time-dependent model |
| 140 | `u_coeff.is_complex() != output_is_complex`->`<` (1) | gap | only real targets into a complex basis are tried; complex targets into a real-output basis must also raise |
| 140 | `!=`->`is not` (1) | equivalent | bools are singletons |
| 144 | error text `'complex' if not output_is_complex` (1) | equivalent | wording of the message only; the tests match `must match the basis coeff_dtype` and the text is not a contract |
| 171 | `values.dtype == regressor.dtype`->`is` (1) | equivalent | torch dtypes are singletons |
| 173 | `itemsize >`->`>=` (1) | dead | equal width with differing dtype needs float16 against bfloat16, which neither basis coefficient precision (complex64/128 -> float32/64) nor the tested regressors produce |
| 194 | `u_coeff_targets.is_complex() !=`->`<` (1) | gap | same as 140, for `test` (`test_on_real_targets` covers one direction) |
| 194 | `!=`->`is not` (1) | equivalent | bools are singletons |
| 213 | `time_shape = shape[1]`->`shape[0]` (1) | gap | metric values of a time-dependent `test` are never asserted; with n != nt the grouping `unflatten(0, (-1, time_shape))` changes the "function value" metrics. Compare against `get_metrics` of a hand-computed inverse transform with n != nt |
| 232 | `if self.complex_funcs`->`if not` (1) | gap | `complex_funcs=True` and `False` runs are smoke-only; assert the "function value" metrics equal an independent computation (complex targets with non-zero imaginary part, interleaved real/imag) |
| 256 | `inverse` `epochs=100`->`101` (1) | equivalent | default undocumented; notebooks and every test pass `epochs` explicitly |
| 258 | `inverse` `gain=0.2`->`1.2` (1) | equivalent | default undocumented; notebooks pass `gain` explicitly. (`inverse` 0.2 vs `inverse_coeff` 0.05 differ, noted, not a survivor) |
| 307 | `generator is None`->`is not` / `not ... is None` (2) | gap | determinism with no generator, and honouring a supplied one, is never asserted |
| 308 | seed `42`->`43` (1) | gap | the default draw is never compared; with `epochs=0` the result is `randn(seed 42) * gain` |
| 310 | `* gain`->`% gain` (1) | gap | same; no test pins the initial guess or `gain` (anchor test passes its own generator but with 800 epochs, which erases the scale) |
| 322 | `requires_grad_(False)`->`True` (1) | gap | returned tensor's `requires_grad` never checked; assert it is False for both real and complex returns |

### Gaps

- 122: non-time-dependent `forward` with an explicit `domain` that differs from `config.domain` equals the basis evaluated on that domain.
- 140, 194: complex targets into a real-output basis raise ValueError in `train` and in `test`.
- 213: time-dependent `test` with n != nt matches `get_metrics` on an independently inverse-transformed field.
- 232: `test` with `complex_funcs` True and False matches an independent "function value" computation.
- 307, 308, 310: `inverse_coeff(u, epochs=0)` with `generator=None` equals `randn(f_shape, Generator(seed 42)) * gain`, twice in a row; with a supplied seed-s generator it equals the draw from that seed.
- 322: `inverse_coeff(...).requires_grad` is False (real and complex output).
