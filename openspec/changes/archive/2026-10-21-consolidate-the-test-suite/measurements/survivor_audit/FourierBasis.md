## SpectralSVR/basis/FourierBasis.py (101 survivors)

The file has no `from __future__ import annotations`; no survivor here is an annotation. Tally: 49 gap, 42 equivalent, 7 dead, 3 unsure, 0 deferred.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 77 | `i=0` to `i=1` (1) | gap | `__call__` default `i` only matters with `n>0`; no test calls the basis with `n>0` and no `i` |
| 83 | `> 0` to `!= 0` (1) | equivalent | `numel()` is never negative |
| 118 | `== 1` to `<= 1` (1) | dead | only a 0-d `x` differs, and both versions raise IndexError (checked: original `IndexError tuple index out of range`) |
| 119 | `-1` to `+1`, to `1` (2) | equivalent | `x` is 1-D inside this branch, so `unsqueeze(-1)` and `unsqueeze(1)` are the same op |
| 121 | `n > 0` to `n > 1` (1) | gap | `n == 1` skips the `coeff[i:i+n]` slice; no test evaluates a sub-range with `n == 1` |
| 123 | `i + n` to `i \| n`, `i ^ n` (2) | gap | identical when `i == 0`; no test evaluates with `i > 0` and `n > 0` (e.g. `i=1, n=1`: slice becomes empty) |
| 124 | `init_len < 1` to `== 1`, `!= 1`, `<= 1`, `> 1`, `>= 1`; `len > 0` to `>= 0`; `init_len < 2` (7) | gap | each switches the "pass valid values" assertion off for a non-empty `coeff`; no test passes `i >= len(coeff)` with `n > 0` (checked: original raises `pass valid values for i and n`) |
| 124 | `len(coeff) > 0` to `!= 0` (1) | equivalent | `len` is never negative |
| 124 | `AddNot`, `True` to `False` (2) | dead | `not` binds to the `True` arm, so only `init_len < 1` (empty `coeff` with `n > 0`) differs; `__call__` refuses empty `coeff`, `evaluate` on an empty one returns an empty result (checked) |
| 133 | `shape[1:]` to `shape[2:]` (1) | gap | only `evaluate(time_dependent=True, domain=None or a single (start, stop) pair)` differs (original accepts both, checked); `__call__` always passes a resolved domain and no test calls the classmethod that way |
| 134 | `> 1` to `!= 1`, `>= 1` (2) | equivalent | a 0- or 1-axis domain still raises AssertionError from `fn`'s own `x.shape[1] == len(domain)` check (checked); only the message text differs |
| 172 | `== 0` to `<= 0` (1) | equivalent | `len` is never negative |
| 172 | `!= 1 + len(modes)` to `is not` (1) | equivalent | small-int identity, same below 257; needs 257+ columns |
| 179 | `!=` to `is not` (1) | equivalent | same small-int identity; needs 257+ axes |
| 188 | `\|` to `+`, `^` (2) | equivalent | `t < start` and `t > stop` cannot both hold (domain validated `stop > start`); bool `+` is logical or (checked) |
| 201 | `/` to `*`, `%`, `**` (3) | equivalent | `ceil - floor` is 0 or 1; weight is `frac` either way, and at integers `0*0`, `0%0` (nan to 0) and `0**0 = 1` all land on a lerp between equal slices (checked on 0, 1, 2.5, 3) |
| 238 | `== len(modes)` to `>=` (1) | gap | an `x` with extra columns is silently accepted and the extra column ignored; no test passes a too-wide `x` to `evaluate` (checked: original raises) |
| 238 | `==` to `<=` (1) | gap | a too-narrow `x` gives IndexError instead of the AssertionError; no test asserts the refusal |
| 238 | `==` to `is` (1) | equivalent | small-int identity |
| 239 | `shape[1]` to `shape[2]` in message (1) | gap | building the message raises IndexError instead of AssertionError; killed by the same refusal test as 238 |
| 259 | `unsqueeze(-1)` to `(+1)`, `(1)` (2) | equivalent | `xb[:, d]` is 1-D |
| 259 | `domain_t[d][0]` to `[1]` (1) | equivalent | shifting by the full length `L` changes the phase by `2*pi*k`, k integer; measured 5.7e-15 max difference in double, below the suite's 1e-12 tolerance; differs only by rounding |
| 266 | `len == 1` to `< 1`, `<= 1` (2) | dead | differ only for zero evaluation points, where both versions raise (ValueError vs IndexError); empty `x` is unsupported |
| 266 | `dim=-1` to `+1`, `1`; `blocks[0]` to `blocks[-1]` (3) | equivalent | blocks are 2-D so dim -1 is dim 1; one block so `[0]` is `[-1]` |
| 268 | remove `@staticmethod` on `sum_mul` (1) | gap | every in-repo caller uses `cls.sum_mul`; calling it on an instance (`basis.sum_mul(coeff_flat, b)`) is untested (low value) |
| 291 | `> 1` to `!= 1`, `>= 1` (2) | gap | a 0-d or 1-D `x` to `fn` raises IndexError instead of AssertionError; refusals only test `modes=None` and a wrong mode count |
| 294 | `> 0` to `!= 0` (1) | equivalent | shape entries are never negative |
| 294 | `> 0` to `>= 0`, `shape[1] > 0` (2) | gap | an `x` with zero rows is accepted instead of refused; no test passes one to `fn` (checked: original raises) |
| 300 | `== len(domain)` to `<=`, `>=` (2) | gap | `<=` accepts a longer per-axis domain and silently ignores the extra axes, `>=` turns the refusal into IndexError; no test passes a per-axis domain of the wrong length |
| 300 | `==` to `is` (1) | equivalent | small-int identity |
| 301 | `shape[1]` to `shape[2]` in message (1) | gap | message building raises IndexError; same refusal test as 300 |
| 314 | `dim + 1` to `dim \| 1`, `dim ^ 1` (2) | equivalent | `dim` is the literal 0 here, all give 1 |
| 343 | `unsqueeze(-1)` to `+1`, `1`; `concat dim=0` to `-1` (3) | equivalent | both operands are 1-D, so dim 0 is dim -1, and dim 1 is dim -1 after one `unsqueeze` |
| 347 | `ord = 1` to `ord = 2` (1) | unsure | every in-repo caller passes `ord` explicitly and no spec states the default; a gap only if the default counts as contract |
| 406 | `> 0` to `!= 0` (1) | equivalent | `len` is never negative |
| 406 | `> 0` to `>= 0` (1) | gap | `generate_empty(n, ())` is accepted instead of refused; no test passes empty modes (checked: original raises) |
| 426 | `> 0` to `!= 0` (1) | equivalent | `len` is never negative |
| 426 | `> 0` to `>= 0`, `> -1` (2) | gap | `generate_coeff(n, ())` is accepted instead of refused; not tested (checked: original raises) |
| 483 | `res.start == start` to `<=`, `>=` (2) | gap | with a window starting outside or inside the domain (`res=slice(0.25, 1.0, 8)`, `slice(-0.25, 1.0, 8)`) the FFT would run on the wrong grid; checked the original skips the FFT there; no test compares such a window with a dense reference |
| 484 | `res.stop == stop` to `<=`, `>=` (2) | gap | same, windows ending short or past the domain (`slice(0, 0.5, 8)`, `slice(0, 1.5, 8)`) |
| 487 | `mode == res.step` to `>=` (1) | gap | `res.step < mode` (e.g. `slice(0, 1, 4)` for 8 modes) would take the FFT and truncate the spectrum; no test resamples below the mode count |
| 487 | `==` to `is` (1) | equivalent | same value for modes up to 256; above 256 the FFT branch is skipped for the exact dense matrix, same values to rounding (route only) |
| 490 | `func == "forward"` to `<=` (1) | equivalent | `func` is "forward" or "inverse" and `"inverse" <= "forward"` is False |
| 492 | `func == "inverse"` to `<=`, `>=` (2) | equivalent | reached only when `func == "inverse"`, where both hold |
| 503 | `func == "inverse"` to `<=` (1) | gap | `"forward" <= "inverse"` is True, so a forward transform off the FFT path under an approximate strategy would take the inverse NUFFT; no test runs the forward transform with a forced-approximate strategy (nufft_available() is True here) |
| 503 | `==` to `>=` (1) | equivalent | `"forward" >= "inverse"` is False, `"inverse" >= "inverse"` True: same as `==` for the two literals |
| 504 | `res.step * mode` to `+`, `-`, `/`, `//`, `<<`, `\|`, `^` (7) | gap | only a budget placed between the product and the altered value changes which route runs; no test sets a budget that separates `step*mode` from its neighbours and checks the route (spy on `nufft_evaluate`, or exact versus approximate result) |
| 580 | `is not` to `is`, `AddNot` (2) | equivalent | `transform` is forward only and `_raw_transform` reads `strategy` only when `func == "inverse"`, so a None or swapped strategy is never used |
| 586 | `1 + 0j` to `1 - 0j`, `1 ** 0j`; `*` to `/` (3) | equivalent | all multiply or divide by exactly 1 |
| 586 | `f * (1+0j)` to `f ** (1+0j)` (1) | dead | bit-identical on 10^6 finite float32 and float64 values (checked); differs only on non-finite input (`inf+nanj` vs `inf+0j`) |
| 598 | `cdim - 1` to `>> 1` in `res=` (1) | gap | equal for cdim 1 and 2; needs a transform over three mode axes with a different `res` per axis |
| 599 | `sampling_axes[cdim - 1]` to `% 1`, `>> 1`, `- 2` (3) | gap | `% 1` and `- 2` differ on two mode axes with different schemes, `>> 1` needs three; forward `transform` is never called with per-axis schemes that differ (the per-axis test uses inverse and only toggles `supports_fft`) |
| 600 | `domain[cdim - 1]` to `>> 1` (1) | gap | equal for cdim 1 and 2; needs a three-axis transform with distinct per-axis domains |
| 675 | `dim - 1` to `% 1`, `>> 1`, `^ 1` (3) | gap | all equal 0 at dim 1; a time-dependent field with two space axes and `grad`/`integral` along dim 2 (and 3 for `>> 1`) with a value check; the state machine only compares shapes |
| 678 | `i != dim` to `is not` (1) | equivalent | small-int identity, needs 257+ axes |
| 687 | `time_size - 1` to `^ 1` (1) | gap | equal for odd `time_size`; the time-grad and time-integral tests use `nt = 17` and `201`, both odd; an even `nt` changes `dt` |
| 690 | `op == "grad"` to `<=` (1) | equivalent | `"integral" <= "grad"` is False, `"grad" <= "grad"` True |
| 698 | `dim == 0` to `<= 0` (1) | dead | only a negative `dim` on a time-dependent field differs, and negative `dim` is already wrong there (`dim - 1` indexes the wrong axis or raises) |
| 698 | `dim == 0` to `>= 0` (1) | gap | every time-dependent `grad` would use the time finite difference; no test checks the values of `grad(dim=1)` on a time-dependent field (state machine checks shape only) |
| 726 | `n: int = 1` to `2` in `plot_coefficients` (1) | unsure | `Basis.plot` documents `n` default 1, `plot_coefficients` does not; smoke tests never count the drawn lines; a gap only if the default counts as contract |
| 729 | `legend: bool = True` to `False` (1) | unsure | same: `Basis.plot` documents the default, `plot_coefficients` does not; no test inspects the legend |

### Gaps

- 77: `basis(x, n=1)` equals evaluating coefficient row 0 alone.
- 121, 123: `evaluate(coeff, x, i=1, n=1)` equals `evaluate(coeff[1:2], x)`.
- 124: `evaluate(coeff, x, i=len(coeff), n=1)` raises AssertionError "pass valid values", and `i=len(coeff)` on a single-function coeff does too.
- 133: `evaluate(td_coeff, x, t=t, time_dependent=True)` with `domain=None` and with a single pair equals the explicit full-axis domain.
- 238, 239: `evaluate` with `x` one column too wide and one too narrow raises AssertionError.
- 268: call `sum_mul` on an instance.
- 291: `fn` with a 1-D and a 0-d `x` raises AssertionError.
- 294: `fn` with a zero-row `x` raises AssertionError.
- 300, 301: `fn` with a per-axis domain one axis too long and one too short raises AssertionError.
- 406, 426: `generate_empty` and `generate_coeff` with `modes=()` raise AssertionError.
- 483, 484: `inv_transform` with `res` windows starting before, starting after, ending before and ending after the domain matches a dense reference.
- 487: `inv_transform` with `res.step` below the mode count matches a dense reference.
- 503: forward `transform` with `PeriodicUniform(supports_fft=False)` and a forced-approximate strategy equals the exact result.
- 504: a strategy whose budget separates `res.step * mode` from `res.step + mode` etc. selects the route the product dictates (spy on `nufft_evaluate`).
- 598, 599, 600: forward `transform` of a three-axis field with different per-axis `res`, sampling (e.g. `ClosedUniform` on one axis) and domain matches a per-axis loop.
- 675: time-dependent `grad` and `integral` along space axis 2 of a field with two space axes match the analytic derivative.
- 687: time `grad` and `integral` on a field with an even `time_size` match the analytic result.
- 698: time-dependent `grad(dim=1)` matches the analytic space derivative, not the time finite difference.
