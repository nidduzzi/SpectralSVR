## SpectralSVR/problems/antiderivative.py (9 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 27 | default `u0 = 0.0` to `1.0` (1) | gap | every test that reaches generation passes `u0` explicitly; the only call without it (`test_antiderivative_rejects_bad_sizes`) raises before `u0` is read |
| 36 | `m > 0` to `m > 1` (1) | gap | `modes=1` is valid (checked: `generate(FourierBasis, 2, 1)` succeeds) and no test uses it; the bad-size test only uses sizes <= 0 |
| 46 | `and` to `or` (1) | dead | `basis.generate` and `grad` always return a non-None coeff, so the assertion cannot fail under either operator |
| 53 | `coeff[:, 0]` to `coeff[:, 1]` (1) | dead | inside the `# pragma: no cover` real-coeff branch; coeff is complex on every reachable path |
| 54 | `or` to `and` (1) | equivalent | `isinstance(u0, float) and isinstance(u0, int)` is never true so floats and ints take the final `else`, which assigns `u0` straight into the complex64 coeff; the original rounds through `torch.tensor(u0 + 0j)` (complex64) to the same value (generate never yields complex128: the `coeff_dtype` class attribute and kwargs do not reach it) |
| 56 | `u0 + 0j` to `u0 - 0j` (1) | equivalent | imaginary part is -0.0 instead of +0.0, which compares equal (`torch.equal(x+0j, x-0j)` is True) |
| 67 | `is not` to `is`, add `not` (2) | gap | both skip `residual.coeff[:, 0].mul_(0)`; the DC coefficient of `u.grad() - ut` is 0 in every test because `ut` is the derivative of a zero-mean field, so zeroing it changes nothing; no test supplies `ut` with a nonzero mean |
| 75 | `grid[1,0] - grid[0,0]` to `+` (1) | gap | every `residual` test uses a domain starting at 0 (`(0.0, length)`), where `x1 + x0 == x1 - x0`; no test uses a domain with start != 0 |

Gaps
- Line 27: `generate(FourierBasis, n, m)` with no `u0` gives `u.coeff[:, 0] == 0`.
- Line 36: `generate(FourierBasis, n, 1)` succeeds and returns `(n, 1)` coefficients.
- Line 67: `spectral_residual(u, ut)` with `ut` having a nonzero DC coefficient returns a residual whose `coeff[:, 0]` is exactly 0.
- Line 75: `residual(u, ut)` on a field with domain start != 0 (e.g. `(1.0, 3.0)`) matches the analytic derivative.
