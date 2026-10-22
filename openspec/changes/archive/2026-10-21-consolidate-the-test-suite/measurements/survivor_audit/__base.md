## basis/__base.py (99 survivors)

`__base.py` has no `from __future__ import annotations`, so annotations are evaluated at definition; annotation-only mutants below are equivalent for other reasons. The tests that cover this file are the `BasisLifecycle` state machine and its pinned replay (`tests/basis/test_fourier_basis.py`) and the plot smoke tests; they assert shapes, not values, for time interpolation and evaluation mode.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 22 | `if not TYPE_CHECKING` (1) | equivalent | the guarded names are used only inside string annotations; the runtime imports have no effect |
| 49 | `==` to `<=`, `>=` (2) | gap | `FourierBasis.transform(x2d, domain=((0.,1.),))` and `domain=((0.,1.),)*3` raise "domain should have the same dimension as modes" (checked) and no test passes a wrong-length domain |
| 49 | `==` to `is` (1) | equivalent | axis counts are small ints, identical under `is` |
| 53 | `strict=True` to `False` (1) | equivalent | line 49 has already enforced equal lengths |
| 60 | `==` to `<=`, `>=` (2) | gap | `transform(x2d, res=(slice(0,1,8),))` raises "expected res of length 2" (checked); no test passes a wrong-length `res` tuple |
| 60 | `==` to `is` (1) | equivalent | small ints |
| 96, 97 | `complex_funcs` / `time_dependent` default False to True (2) | dead | `FourierBasis.__init__` always forwards both; only a second `Basis` subclass omitting them reaches the default, none exists |
| 124 | `*,` to `/,` in `from_config` (1) | gap | no test calls `from_config`; the keyword-only contract of `time_dependent` / `complex_funcs` is unpinned (`from_config(c, cfg, True)` should raise `TypeError`) |
| 144 | remove `@staticmethod` on abstract `default_sampling` (1) | equivalent | `FourierBasis` overrides it as a staticmethod and the abstract body is never called |
| 145 | remove `@abc.abstractmethod` on `default_sampling` (1) | gap | no test defines a `Basis` subclass lacking it and expects `TypeError` on instantiation |
| 157 | `numel() == 0` to `<= 0` (1) | equivalent | `numel` is never negative |
| 160 | `coeff.ndim >= 2` to `>= 1` (1) | gap | `FourierBasis(torch.zeros(4, dtype=torch.complex64))` raises "at least a two dimensional" (checked) and no test builds a 1-D coefficient |
| 209 | `numel() == 0` to `<= 0` (1) | equivalent | `numel` is never negative |
| 267 | `strict=True` to `False` in `_grid_from_schemes` (1) | gap | `get_values_and_grid(res=(s,s,s), evaluation_mode="basis")` on a 2-axis basis raises `ValueError: zip() ... longer` (checked); the inverse-transform path is caught earlier by line 60, the basis path only here; mutant behaviour not run, expected to fail with another error type |
| 280 | `n > 0` to `>= 0` (1) | gap | `n=0` currently evaluates all functions (`get_values(n=0)` checked); mutant evaluates none; no test passes `n=0` |
| 280 | `n > 0` to `n > 1` (1) | gap | differs for `n=1` on a basis with more than one function; the pinned selection uses `count=2` and the fuzz never reaches `count=1, len>1` |
| 281 | `i : i + n` to `i \| n`, `i ^ n` (2) | gap | pinned `start=1, count=2` gives `1\|2 = 1^2 = 3 = 1+2`, so blind; `start=1, count=1` (or `2, 2`) kills it |
| 333 | `len(res) > 1` to `!= 1`, `>= 1` (2) | gap | `res=()` and `res=(s,)` on a time-dependent basis in `"basis"` mode raise "res list should be more than one element" (checked); no test passes them |
| 362 | `ndim + 1` to `- * / // % ** >> << \| & ^`, `+ 2` (12) | gap | only used when `res` is a single `slice` on a time-dependent basis; `read_values_at_a_resolution` passes a single slice only for time-independent 1-axis bases; `get_values(res=slice(...))` on a time-dependent basis returns a `(n, m, m)` result (checked) |
| 370 | `strict=False` to `True` (1) | gap | differs only for a stated domain whose axis count differs from the coefficients; original truncates then fails on line 60 with `AssertionError`, mutant raises `ValueError`; no test constructs such a basis |
| 380 | `i=0` default to `1` (1) | gap | `get_values(n=1)` without `i` should return function 0; `i` is ignored while `n<=0`, so only an explicit `n>0` without `i` on a multi-function basis tells them apart; tests always pass `i=start` |
| 381 | `n=-1` to `~1`, `not 1`, `-2` (3) | equivalent | any `n <= 0` takes the "all functions" branch in both `__base` 280 and `FourierBasis.evaluate` |
| 402 | `evaluation_mode == "auto"` to `<=` (1) | equivalent | the valid modes `basis` and `inverse transform` sort after `"auto"`, so `<=` is `==` for supported inputs |
| 402 | `== "auto"` to `>=` (1) | gap | `"basis"` becomes the preferred mode (`inverse transform`); the two paths agree to 1.4e-6 in float32 (checked), and the test compares shapes only; needs a spy on `__call__` |
| 407 | `== "inverse transform"` to `<=` (1) | gap | `"basis"` sorts before it and would take the inverse-transform path; same blind spot as 402 |
| 407 | `== "inverse transform"` to `>=` (1) | equivalent | `"basis"` is below it and `"inverse transform"` equal, so `>=` is `==` for supported inputs |
| 472, 473 | `i`, `n` defaults on abstract `__call__` (2) | equivalent | abstract stub with `pass`; `FourierBasis.__call__` declares its own defaults |
| 494 | remove `@overload` (1) | equivalent | the later `def evaluate` rebinds the name, the stub never exists at runtime |
| 503, 504, 519 | `n=0` to `1`, `Literal[True]` to `Literal[False]`, `Literal[False]` to `Literal[True]` in `@overload` stubs (3) | equivalent | overload stubs are replaced by the final `evaluate`; evaluated annotations stay valid |
| 694, 709 | `dim=0` to `1` on abstract `grad`, `integral` (2) | equivalent | docstring-only abstract bodies; `FourierBasis` overrides with its own `dim=0` |
| 725 | `ord=1` to `2` on `derivative_eigenvalues` (1) | equivalent | base body returns `None` and ignores `ord` |
| 743 | `2.0 / 3.0` to `+ - * // % **`, `3.0 / 3.0` (7) | equivalent | base `dealias_mask` returns `None` regardless of `fraction`; `FourierBasis` overrides it |
| 793 | `n: int = 1` to `2` on abstract `plot_coefficients` (1) | equivalent | docstring-only; `FourierBasis.plot_coefficients` has its own default |
| 810 | `complex_scatter=False` to `True` default (1) | gap | tests always pass `complex_scatter` explicitly; `basis.plot()` on a complex field should draw lines |
| 812 | `legend=True` to `False` default (1) | gap | tests never pass `legend` and never read a legend |
| 853 | `grid(res=200)` to `201` (1) | gap | no test calls `Basis.grid()` without `res`; every call states it |
| 874 | `resize_modes(rescale=True)` to `False` (1) | gap | no method-level test calls `resize_modes` with the default and checks values (`test_fuzz_resize_modes_sets_target` checks the mode count; the problems test compares two default calls against each other); the rescale is anchored only on the `utils.resize_modes` function |
| 909 | `linspace(0,1,nt) * (T-1)` to `/ // % **`, `linspace(1,1,nt)` (5) | gap | `resize_through_time` asserts `modes` and `time_size` only; interpolated time values are never checked |
| 910 | `time_size - 1` to `% 1`, `>> 1`, `& 1`, `- 2` (4) | gap | same as 909 |
| 920 | `perturb(std_ratio=0.1)` to `1.1` (1) | gap | the one perturb test passes `std_ratio` and only checks shape and `not allclose` |
| 940 | `values + rand(...)` to `- * / **` (4) | gap | noise is symmetric, so the sign is invisible to a shape and `not allclose` check; needs a deterministic `rand_func` and `values + noise * ratio * std` as oracle |
| 944 | `if self._complex_funcs` negated (1) | gap | noise dtype choice (complex for complex fields, real otherwise); assert the dtype passed to `rand_func`, or that a real basis stays real |
| 947, 948 | `* std_ratio`, `* values.std()` to `+ - / // % **` (6 each) | gap | same as 940: the scale is never compared with an expected value |
| 980 | `dtype is torch.bool` to `== torch.bool` (1) | equivalent | `torch.dtype` has no custom `__eq__`; numpy and torch dtypes compare False to `torch.bool` (checked), the pinned numpy and tensor slice counts survive |
| 981 | `ndim != 0` to `> 0` (1) | equivalent | `ndim` is never negative |
| 992 | `requested != time_size` to `is not` (1) | equivalent | below 257 equal ints are the same object; differs only when `time_size > 256` (a request equal to its own count of e.g. 300 slices would raise); call it a gap only if more than 256 slices is a realistic case here |
| 1013 | `ndim - 1` to `>> 1`, `^ 1` (2) | equivalent | this branch holds `ndim >= 1`; the value is used only in `< 1` and in the message, and `1-1`, `1>>1`, `1^1` all give 0, while `ndim >= 2` is never below 1 in any variant |
| 1028 | `strict=True` to `False` (1) | gap | differs only for a stated domain of the wrong axis count: original raises `ValueError` from `zip`, mutant truncates; no test converts such a basis |
| 1080 | `domain[0]` to `domain[1]` (1) | gap | `to_time_independent` takes the time interval from axis 0; every test domain is `(0,1)` on all axes or sets spans without comparing node times; use time `(0, 2)` with space `(0, 1)` and check the sampled times |

### Gaps

- 49, 60: `FourierBasis.transform` refuses a domain or a `res` tuple whose length differs from the field's axis count (`AssertionError`).
- 124: `from_config(coeff, config, True)` raises `TypeError`.
- 145: a `Basis` subclass without `default_sampling` cannot be instantiated.
- 160: a 1-D complex coefficient raises "at least a two dimensional".
- 267: `res` of the wrong length with `evaluation_mode="basis"` raises `ValueError`.
- 280: `get_values(n=0)` returns all functions; `get_values(i=1, n=1)` and `get_values(n=1)` select function 1 and 0 of a multi-function basis.
- 281: pin `read_a_selection_of_functions(start=1, count=1)` and `(start=2, count=2)` with 4 functions.
- 333: time-dependent basis, `"basis"` mode, `res=()` and `res=(s,)` raise the "more than one element" assertion.
- 362: `get_values(res=slice(...))` on a time-dependent basis, in both modes, returns shape `(n, m, m)`.
- 370, 1028: a basis with a stated domain of the wrong axis count refuses (`AssertionError`/`ValueError`) in `get_values` and `to_time_dependent`.
- 380: `get_values(n=1)` equals the first function of a 3-function basis.
- 402, 407: spy that `evaluation_mode="basis"` calls `__call__` and `"inverse transform"` does not; `"auto"` follows `prefered_evaluation_mode`.
- 810, 812, 853: `basis.plot()` defaults (lines, legend present) and `Basis.grid()` has 200 points.
- 874: `resize_modes(target)` default equals `rescale=True` and differs from `rescale=False`; the represented function is preserved.
- 909, 910: resizing a time-dependent field to a target with other slices reproduces a field linear in time at the new slice times (analytic), and equal slice counts is the identity.
- 920, 940, 944, 947, 948: `perturb` with a deterministic `rand_func` returning ones gives `values + 0.1 * std(values)` by default and `ratio * std` for a given ratio; real basis stays real, complex gets complex noise.
- 1080: `to_time_independent` with time domain `(0, 2)` and space `(0, 1)` samples at the time interval's nodes.
