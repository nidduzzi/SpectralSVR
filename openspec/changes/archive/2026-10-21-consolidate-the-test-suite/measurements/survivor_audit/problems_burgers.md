## SpectralSVR/problems/burgers.py (169 survivors)

Verdict counts: 66 gap, 94 equivalent, 9 dead. Rows marked "message" mutate only the text of an error whose type and prefix the tests (or the contract) already fix; no spec scenario pins the wording beyond the regexes in the tests. No `from __future__ import annotations` in this file, but no annotation-only mutant survived here.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 22 | `frozen=True` to `False` on `SpectralForcing` (1) | gap | no test assigns to a `SpectralForcing` (the other forcing dataclasses are killed, this one is not) |
| 39 | `+ 0j` to `- 0j` (1) | equivalent | imaginary part -0.0 vs +0.0, which compare equal and feed the same linear transform (`torch.equal(transform(x+0j), transform(x-0j))` checked True) |
| 81 | default `nu = 0.01` to `1.01` (1) | gap | every test that reads values passes `nu`; the tests that omit it (`honors_domain_start`) assert only the domain |
| 119 | default `time_domain` start 0 to -1 (1) | gap | tests that omit `time_domain` assert only shapes, NaN-freedom or the space/time domain they passed; none asserts the default `(0, 1)` |
| 122 | `n > 0` to `!= 0`, `>= 0`, `> -1` (3) | gap | the sample-count refusal is tested only for Antiderivative; no Burgers test passes `n = 0` or `n < 0` |
| 123 | zero-iteration loop over `modes` (1) | gap | the per-mode validation is never exercised for Burgers (no test passes a nonpositive mode count) |
| 124 | `m > 0` to `!= 0`, `>= 0`, `> -1` (3) | gap | as line 123: `m = 0` and `m < 0` are never passed to `Burgers.generate` |
| 132 | `nt > 1` to `nt > 2` (1) | gap | the refusal test uses `step <= 1` only; `nt = 2` is valid (checked: generation succeeds with `(1, 2, 8)` coeffs) and never used |
| 138 | message operands `t[1] - t[0]` to other operators, `t[0]` to `t[2]` (12) | equivalent | message of an assert no test triggers; text only (the assert is reachable: `time_domain=slice(1000., 1001., 200)` fails it because the grid is float32), wording not a contract |
| 143 | `u0 == "random"` to `<=`, `>=` (2) | gap | any other string compares True under one of them (`"abc" <= "random"`), so `u0="abc", f="random"` is silently manufactured instead of refused; no test passes a string other than "random" |
| 145 | `f == "random"` to `<=`, `>=` (2) | gap | same for `f`: `f="zzz"` with `u0="random"`; no test passes another string |
| 168 | `flatten(0, 1)` to `flatten(1, 1)` (1) | gap | `flatten(1, 1)` is a no-op and `transform` on a 3-D `(n, nt, ns)` array transforms two axes (checked: result differs from the per-slice transform); the manufactured branch's time-dependent output is checked only for shape, NaN, domain, never against the field's values |
| 262 | `len(modes) != 2` to `< 2`, `> 2` (2) | gap | the numerical branch's "1D space" refusal is never tested (no test passes a 1-tuple or 3-tuple `modes` with a non-manufactured pair); `< 2` lets 3 modes through, `> 2` lets 1 mode reach `modes[1]` |
| 265 | `domain[1][1] - domain[1][0]` to `+` (1) | gap | numerical tests use space domains starting at 0, where sum and difference agree; `honors_domain_start` uses other starts but asserts only the domain |
| 269 | `d1 is None or d2 is None` to `and` (1) | dead | differs only for a basis whose first derivative is diagonal and second is not; FourierBasis (the only concrete basis) returns both, the base class returns neither, so the branch is reached only with both None, where `and` and `or` agree |
| 278 | `is not` to `is`, add `not` (2) | gap | both drop the 2/3 de-aliasing mask in the solver; measured on `u0 = 0.5 sin(2 pi 8 x)`, ns = 32: coefficients above the mask stay at 3.5e-17 with the mask and reach 0.25 without it; tests pass the same mask to every solver or use band-limited manufactured fields where aliasing is negligible |
| 278 | `1.0` to `2.0` (1) | dead | the `else` is taken only when `dealias_mask` is None, which FourierBasis never returns; needs a stub diagonal basis without a mask |
| 280 | `u0 == "random"` to `<=`, `>=` (2); `and` to `or` (1) | gap | an unrecognised string `u0` (numeric `f`) is silently turned into a random initial condition instead of raising `NotImplementedError`; no test passes one |
| 283 | `+ 0j` to `- 0j` (1) | equivalent | signed zero in the imaginary part, as line 39 |
| 287 | `field.ndim == 1` to `<= 1` (1) | gap | a 0-d tensor `u0` is refused by the original (`AssertionError`, checked) and broadcast by the mutant; no test passes a 0-d tensor |
| 289 | `field.shape == (n, ns)` to `<=`, `>=`, `is not` (3) | gap | no test passes an `u0` of the wrong shape: `is not` makes the assert always pass, `<=` passes any lexicographically smaller shape, `>=` any larger one |
| 292 | `+ 0j` to `- 0j` (1) | equivalent | signed zero, as line 39 |
| 315 | remove `@staticmethod` on `_forcing_at` (1) | dead | its only caller uses `Burgers._forcing_at(...)` through the class, where a plain function behaves identically; only an instance call (unsupported for a private helper) differs |
| 339 | `+ 0j` to `- 0j` (1) | equivalent | signed zero, as line 39 |
| 358 | `spectrum.shape != (n, ns)` to `>` (1) | gap | the wrong-shape test returns `(n, ns + wrong)`, lexicographically larger; a smaller shape (`(n, ns - 1)`, `(n - 1, ns)`) is never passed and would raise a different error type from the later add |
| 402 | `domain[0]` to `domain[1]` for `time_interval` (1) | gap | every forcing-field test uses time `(0, 1)` and space `(0, 1)`, so the two intervals coincide; a time interval different from the space interval (e.g. `(0, 2)` vs `(0, 1)`) with a matching field would be refused by the mutant |
| 403 | `ndim != 3` to `< 3` (1) | gap | the test passes a 2-axis field only; a 4-axis coefficient array (ndim 4) is never passed |
| 406 | `ndim - 1` to other operators and `- 2` (12) | equivalent | message: the axis count inside the text; the test pins the prefix `must span .time, space.` only |
| 414 | `field.domain[0] != time_interval` to `>` (1) | gap | the test uses `(offset, offset + 1)` with positive offset, always lexicographically greater than `(0, 1)`; a field with a smaller time domain (`(-0.5, 0.5)`, `(0, 0.5)`) is never passed |
| 416 | `domain[0]` to `domain[1]` in the message (1) | equivalent | message text only |
| 419 | `field.domain[1] != space_interval` to `>` (1) | gap | the test uses `(0, 1 + stretch)`, always greater than `(0, 1)`; a smaller or earlier-starting space domain is never passed |
| 442 | `dim=-1` to `+1`, `1` (2) | equivalent | `torch.stack` of two 1-D tensors: `dim=-1` and `dim=1` are the same axis (checked) |
| 500 | `strict=True` to `False` (1) | equivalent | `bands` is built with one entry per `modes` entry, so the lengths always agree |
| 503 | `strict=True` to `False` (1) | equivalent | same as line 500 |
| 543 | `not manufactured and band is not None` to drop `not` and double `not` (2) | gap | no test calls `generate` with a `band` (only `mms_solution` directly): `generate(u0="random", f="random", band=...)` must work and a numerical call with `band` must raise "nothing to limit"; both mutants flip these |
| 562 | `modes // 2` to `+ 2`, `- 2`, `* 2`, `/ 2`, `% 2`, `** 2`, `>> 2`, `<< 2`, `\| 2`, `& 2`, `^ 2`, `// 3` (12) | gap | no Burgers test passes `WholeGrid` (only `utils` tests use it); `highest_frequency(WholeGrid(), modes)` is never evaluated and `% 2`, `& 2` would even let `WholeGrid` through `_check_band` at small grids |
| 564 | remove `@classmethod` on `_check_band` (1) | dead | the only caller is `self._check_band(...)`, so `cls` binds the instance and `cls.widest_manufactured_band`/`highest_frequency` resolve the same staticmethods; only a class-level call differs |
| 572 | `2 * c + 1 > modes` mutated by `+` to `-`, `*`, `/`, `//`, `%`, `**`, `>>`, `<<`, `\|`, `&`, `^`; `*` to `>>`, `&`; `2` to `3` (14) | equivalent | the first check is subsumed by the second: when `2c + 1 > modes`, `c > modes // 6` always holds, so a ValueError is raised either way (enumerated over modes 1..79, cutoffs 0..44: same raise/no-raise); only which message appears differs |
| 572 | `*` to `+`, `-`, `\|`, `^` (4); `>` to `==`, `>=`, `is` (3) | gap | over-refuse the smallest valid grids: modes 1 and 2 with cutoff 0 (mutants `2+c+1`, `2-c+1`, `2\|(c+1)`, `2^(c+1)` raise at modes 1, 2; `==`, `>=`, `is` raise at modes 1); `mms_solution` with `(m, m)` is accepted at m = 1..5 in the original and no test uses `m < 6` |
| 572 | add `not` before `isinstance` (1) | gap | `WholeGrid` has no `cutoff`, so the mutant raises `AttributeError` instead of the intended `ValueError`; no test passes `WholeGrid` to `mms_solution`/`_check_band` |
| 575 | `2 * c + 1` in the message (23) | equivalent | message text only (tests match the regex `past the\|more than`) |
| 582 | `2 * stated`, `modes / 3`, `2 *` in the message (23) | equivalent | message text only |
| 603 | `[-1]` to `[+1]`, `[1]` (2) | dead | `u` is a 2-axis field, so index 1 and -1 agree; a 3-axis `u` is unsupported here |
| 603 | `[-1]` to `[~1]`, `[not 1]`, `[-2]` (3) | gap | each selects the time-axis mode count; every test calling `spectral_residual` uses square `(m, m)` grids, so time and space counts coincide |
| 604 | `lengths[-1]` to `[+1]`, `[1]` (2) | dead | as line 603 |
| 604 | `lengths[-1]` to `[~1]`, `[not 1]`, `[-2]` (3) | gap | each selects the time length; the independent oracle tests use domain `(0,1)x(0,1)` and the MMS tests with unequal lengths are self-consistent (forcing built by the same function) |
| 611 | `1.0` to `2.0` (1) | dead | the `else` is taken only when `dealias_mask` is None, which FourierBasis never returns |
| 624 | default `res = 1000` to `1001` (1) | gap | every Burgers `residual` test passes `res`; none uses the default |
| 628 | `grid[1,0,0] - grid[0,0,0]` to `+` (1) | gap | every residual test uses a time domain starting at 0, where `t1 + t0 == t1 - t0` |
| 629 | `grid[0,1,1] - grid[0,0,1]` to `+` (1) | gap | every residual test uses a space domain starting at 0 |
| 629 | `grid[0,1,1]` to `grid[1,1,1]` (1) | equivalent | the sampled grid is a tensor product, so the space coordinate at space index 1 is the same for every time index (checked: both 0.1) |
| 640 | time-dependent branch `time_periodic = False` to `True` (1) | gap | periodic and non-periodic order-1 stencils agree in the interior and differ only at the two time end samples; every time-dependent test trims `b = 6..8` boundary rows, and the others check only NaN-freedom |
| 644 | `axis_schemes[0]` to `[1]` for `time_periodic` (1) | gap | both axes are periodic in every test, so the scheme index is invisible; a per-axis sampling `(ClosedUniform(), PeriodicUniform())` on a non-time-dependent field separates them (touches the deferred ClosedUniform defect, see unsure) |

Gaps
- Line 22: assigning to `SpectralForcing(...).at_time` raises `FrozenInstanceError`.
- Line 81: `generate(...)` without `nu` equals `generate(..., nu=0.01)` (coefficients `torch.equal`).
- Line 119: `generate(...)` without `time_domain` has `u.domain[0] == (0.0, 1.0)` and 200 time points.
- Lines 122, 123, 124: `generate` with `n = 0`, `n = -1`, and a mode count of 0 or -1 in either axis raises `AssertionError`.
- Line 132: `time_domain=slice(0, 1, 2)` is accepted and returns two time slices.
- Lines 143, 145, 280: a string `u0` or `f` other than "random" (e.g. "abc") is refused for every branch combination.
- Line 168: manufactured `generate` with `time_dependent_coeff=True` reproduces the underlying field's values at the time grid (compare `inv_transform` of the output with the `mms_solution` field sampled at the grid).
- Line 262: numerical `generate` with `modes` of length 1 and of length 3 raises `NotImplementedError`.
- Line 265: numerical `generate` on space domain `(1.0, 3.0)` matches the translated `(0.0, 2.0)` run or the Cole-Hopf closed form on that period.
- Line 278: with band-limited `u0 = 0.5 sin(2 pi 8 x)` at ns = 32, coefficients above the 2/3 mask stay below 1e-12 (measured 3.5e-17; 0.25 without the mask).
- Line 287: a 0-d tensor `u0` raises `AssertionError`.
- Line 289: an `u0` of shape `(n, ns - 1)`, `(n - 1, ns)`, `(n, ns + 1)` and `(n + 1, ns)` each raises `AssertionError`.
- Line 358: a `SpectralForcing` callable returning `(n, ns - 1)` and `(n - 1, ns)` raises `ValueError` "forcing callable must return".
- Line 402: a forcing field on time `(0, 2)` and space `(0, 1)` is accepted when the integration intervals match, and refused when they are swapped.
- Line 403: a forcing field with 4-axis coefficients raises `ValueError` "must span (time, space)".
- Lines 414, 419: a forcing field whose time (space) domain is smaller than or starts before the integration interval raises `ValueError`.
- Line 543: `generate(u0="random", f="random", band=WavenumberCutoff(k))` honours the band; `generate(u0=0.0, f=0.0, band=...)` raises `ValueError`.
- Line 562: `Burgers.highest_frequency(WholeGrid(), modes) == modes // 2` for even and odd `modes`, and `mms_solution(band=WholeGrid())` raises `ValueError` at `modes` 6..40.
- Line 572: `mms_solution` with `(m, m)` for `m` in 1..5 and the default band is accepted; `mms_solution(band=WholeGrid())` raises `ValueError` (not `AttributeError`).
- Lines 603, 604: `spectral_residual` on a non-square field (`nt != ns`) with time length != space length matches an analytic periodic pair.
- Line 624: `residual(u, f, nu)` with the default `res` returns 1000 coefficients per axis.
- Lines 628, 629: `residual` on a field with time and space domain starts != 0 matches the analytic residual.
- Line 640: `residual` on a time-dependent field read at the first and last time rows matches an analytic residual (one-sided stencil, not wrapped).
- Line 644: `residual` on a non-time-dependent field sampled with `(ClosedUniform(), PeriodicUniform())` matches the analytic residual (time axis non-periodic).
