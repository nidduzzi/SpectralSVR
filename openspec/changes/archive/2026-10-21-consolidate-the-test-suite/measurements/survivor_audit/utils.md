## SpectralSVR/utils/__init__.py (98 survivors)

The file has no `from __future__ import annotations`, but no survivor here changes a `|` annotation; the two decorator survivors (267, 331) are discussed per row. Tally: 37 gap, 50 equivalent, 10 dead, 1 unsure, 0 deferred. A CUDA device is present on this host, which matters for line 51.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 51 | `device.type == present.type` to `>=` (1) | gap | with an accelerator present (cuda here), any type sorting at or after "cuda" (e.g. `torch.device("meta")`) would be re-indexed to `meta:<n>`; `placed(torch.device("meta"))` returned `meta` (checked); no test uses a device type other than cpu/cuda; equivalent on a host with no accelerator |
| 83 | `source.index == placement.index` to `>=` (1) | gap | source index above the placement index (`cuda:1` source, `cuda:0` work) would be accepted; needs two stated indices, which a stub carrying a `.device` attribute supplies without hardware; the tests only use `cuda` and `cuda:0` |
| 83 | `==` to `is` (1) | equivalent | small-int identity for device indices |
| 123 | `coeff_dtype is not None` to `is None`, `AddNot` (2) | gap | an explicit `coeff_dtype` is then ignored (checked: `to_complex_coeff(float32, coeff_dtype=complex128)` gives complex128); no test passes `coeff_dtype` to `to_complex_coeff` (grep found none) |
| 218 | `s <= eps` to `==`, `is` (2) | gap | no clamp for `s < eps`: a constant column keeps `s = 0` (checked: original gives `ss = 1e-12`); no test fits a constant column |
| 218 | `s <= eps` to `<` (1) | equivalent | the only case that changes is `s == eps`, and it assigns `eps` to `eps` |
| 221 | `eps=1e-12` to `-0.999999999999` (1) | gap | negative eps never clamps (std is non-negative), same constant-column test as 218 |
| 231 | `i is None` to `is not`, `AddNot` (2) | dead | the name only builds the message of the assertion at 232 or 241, which `_check_consistency` makes unreachable through `transform`/`inverse` |
| 232 | `x.shape[1] == m.shape[1]` to `<=`, `>=`, `is` (3) | dead | `_translate` is only reached after `_check_consistency` has forced the widths equal (checked: with that check removed the assertion fires), so it never fails; only a direct call to the private static distinguishes (`is`: small ints) |
| 241 | `x.shape[1] == s.shape[1]` to `<=`, `>=`, `is` (3) | dead | same as 232 for `_scale` |
| 249 | `xs_is_complex ==` to `<=` (1) | equivalent | a kind mismatch with equal raw width leaves widths unequal after real conversion, and `_translate`/`_scale` raise the same AssertionError (checked); only the message text differs |
| 253 | `xs_dims ==` to `<=`, `>=`, `is not` (3) | equivalent | the per-field width assertions in `_translate`/`_scale` raise AssertionError for any width mismatch (checked with the check removed); only the message differs |
| 267 | remove `@typing.overload` on the first `transform` stub (1) | equivalent | the later `def transform` rebinds the name, so the decorator only informs type checkers |
| 278 | `strict=False` to `True` (1) | equivalent | `_check_consistency` already forces equal lengths of the compared tuples |
| 282 | `strict=False` to `True` (1) | equivalent | same: `ms` has one entry per fitted field |
| 286 | `strict=False` to `True` (1) | equivalent | same: `ss` has one entry per fitted field |
| 288 | `strict=False` to `True` (1) | equivalent | same lengths |
| 288 | loop body replaced by `for x, m in []` (1) | gap | skips the `nan_to_num_` in `transform`; no test passes NaN or inf to `transform` (checked: original maps a NaN to `ms`, e.g. 0.8274); the identical loop in `inverse` at 319 did not survive, and I could not find which test kills it. Pins the behaviour that NaN becomes the mean in the standardised space, which may not be the intended oracle (0 would be the mean) |
| 289 | `for dim in range(...)` replaced by `[]` (1) | gap | same NaN input to `transform` as 288 |
| 290 | `m[0, dim]` to `m[-1, dim]` (1) | equivalent | `m` has shape `(1, d)` (`keepdim=True`), so row -1 is row 0 |
| 294 | `strict=False` to `True` (1) | equivalent | same lengths |
| 296 | `xs_out[0]` to `xs_out[-1]` (1) | equivalent | tensor input gives a one-element tuple |
| 313 | `strict=False` to `True` (1) | equivalent | same lengths, `inverse` checks consistency too |
| 317 | `strict=False` to `True` (1) | equivalent | same lengths |
| 331 | remove `@staticmethod` on `StandardScaler.load` (1) | gap | tests call `StandardScaler.load(path)` on the class, which still works; `scaler.load(path)` on an instance is untested (low value) |
| 353 | `shape[0] > 1` to `!= 1` (1) | dead | only a zero-row target differs (`nan` vs ValueError from torchmetrics, checked); `get_metrics` already fails on zero rows in the per-sample statistic, so only a direct call to the private `_safe_r2` reaches it |
| 432 | `metrics is None` to `is not`, `AddNot` (2) | equivalent | `cast` is the identity at runtime, both arms return `results` |
| 435 | `rescale=True` to `False` (1) | gap | the free function's default is never used: tests pass `rescale` explicitly, `Basis.resize_modes` always forwards it, the smoke test checks shape only; `Basis.resize_modes` documents default True. Test: `resize_modes(x, n)` equals `resize_modes(x, n, rescale=True)` |
| 450 | `len == ` to `<=`, `>=` (2) | gap | a target tuple longer or shorter than the mode axes is accepted, zip truncates and the rescale uses the wrong product (silent), or IndexError; no test passes a mismatched target length |
| 450 | `==` to `is` (1) | equivalent | small-int identity |
| 455 | `strict=False` to `True` (1) | equivalent | lengths are asserted equal at 450 |
| 467 | `target_mode // 2` to `/ 2` (1) | equivalent | `torch.tensor([x.5], dtype=torch.long)` truncates toward zero, equal to the floor for non-negative values (checked: `[1.5, 2.0, 2.9]` gives `[1, 2, 2]`) |
| 476 | `target_mode % 2 == 0` to `<= 0` (1) | equivalent | `% 2` is 0 or 1 |
| 477 | `target_mode // 2` to `/ 2` (1) | equivalent | inside the even-`target_mode` branch the quotient is an integer |
| 483 | `target_mode > current_mode` to `!= ` (1) | equivalent | reached only when `target_mode >= current_mode`, where `!=` and `>` agree |
| 483 | `>` to `is not` (1) | gap | equal for mode counts up to 256 (cached ints); for equal counts above 256 the objects differ, so an even-sized no-op resize would enter the pad branch and halve the Nyquist coefficient; counts above 256 are realistic here. Test: `resize_modes(x, x.shape[1:])` with 512 modes returns `x` |
| 493 | `current_mode > 1` to `!= 1`, `>= 1`, `> 0` (3) | equivalent | for `current_mode` 0 or 1 `range(high_start, current_mode)` is already empty (high_start equals current_mode), so the guard is redundant |
| 509 | `current_mode % 2 == 0` to `<= 0` (1) | equivalent | `% 2` is 0 or 1 |
| 511 | `current_mode // 2` to `/ 2` (1) | equivalent | inside the even branch, integer-valued |
| 513 | `current_mode // 2` to `/ 2` (1) | equivalent | same |
| 543 | `2 * cutoff + 1` to `\| 1`, `^ 1` (2) | equivalent | `2 * cutoff` is even for the int `cutoff` (annotated `int`, negatives rejected), so `+ 1`, `\| 1`, `^ 1` agree |
| 614 | `len(shape[1:]) == len(bands)` to `<=` (1) | gap | a band list longer than the axes passes and raises IndexError instead of AssertionError; no test passes a wrong-length list |
| 614 | `==` to `>=` (1) | gap | a band list shorter than the axes passes and silently leaves the remaining axes unbanded |
| 614 | `==` to `is` (1) | equivalent | small-int identity |
| 628 | `zeros((1,))[0]` to `zeros((2,))[0]` (1) | equivalent | both give a scalar zero |
| 634 | `dim: int = 1` to `2` in `interpolate_tensor` (1) | unsure | every caller and test passes `dim=1` explicitly; no doc states the default; a gap only if the default counts as contract |
| 637 | `remainder(1)` to `remainder(2)` (1) | equivalent | only all-integer odd indices change branch, and the lerp branch then has weight 0 (0/0 to nan to 0), returning `x_floor` exactly, the same as `index_select` |
| 646 | `/` to `*`, `%`, `**` (3) | equivalent | `ceil - floor` is 0 or 1, so the weight is the same fraction; at integer indices the weight is 0 or 1 between two equal slices (checked in the 201 analysis of FourierBasis) |
| 674 | `len(t.shape) == 1` to `<=` (1) | gap | a 0-d `t` gives TypeError from `len` instead of AssertionError; no test passes it |
| 674 | `==` to `>=` (1) | gap | a 2-D `t` passes the assertion instead of being refused; no test passes it |
| 677 | `len(t) > 1` to `!= 1` (1) | gap | an empty `t` is accepted and fails later with IndexError; the refusal test only passes a single point |
| 677 | `> 1` to `> 2` (1) | gap | a two-point grid is refused; no test integrates over exactly two points |
| 682 | `j - 1` to `% 1`, `>> 1`, `^ 1` (3) | equivalent | `j` is the literal 1, all give 0 |
| 683 | `strict=False` to `True` (1) | equivalent | `t[:-1]` and `t[1:]` have equal length |
| 687 | `isnan().sum() == 0` to `<= 0` (1) | equivalent | a count is never negative |
| 687 | `== 0` to `>= 0` (1) | gap | the NaN check is disabled; no test uses a right-hand side that produces NaN and expects the AssertionError |
| 782 | `2 * contour_points` to `2 + `, `2 \| `, `2 ^ `, `3 * ` (4) | gap | the complex branch is documented as twice as many nodes; with 32 points every variant (34 or 96 nodes) still converges inside the closed-form tolerance, so only an exact comparison with a reference built from `2 * contour_points` nodes (as `test_etdrk4_real_weights_unchanged` does for the real branch) tells them apart |
| 785 | `mean(dim=-1)` to `+1`, `1` (2) | equivalent | `expr` is 2-D `(n, count)` |
| 791 | `mean(dim=-1)` to `+1`, `1` (2) | equivalent | same |
| 793 | `unsqueeze(-1)` to `+1`, `1` (2) | equivalent | operand is 1-D |
| 807 | `contour_points: int = 32` to `33` (1) | gap | the docstring states default 32; every test uses the default and a 33-node result is within tolerance; test: `etdrk4_solver(...)` equals `etdrk4_solver(..., contour_points=32)` exactly |
| 825 | `t.ndim == 1` to `<=` (1) | gap | a 0-d `t` gives TypeError instead of AssertionError |
| 825 | `==` to `>=` (1) | gap | a 2-D `t` is not refused |
| 825 | `len(t) > 1` to `!= 1` (1) | gap | an empty `t` passes and fails later with IndexError |
| 825 | `> 1` to `>= 1` (1) | gap | a single time point passes the assertion and fails later |
| 825 | `and` to `or` (1) | gap | any of a 0-d, 2-D, or one-point `t` passes |
| 825 | `> 1` to `> 2` (1) | gap | a two-point grid is refused; no test integrates over exactly two points |
| 827 | `8 * eps * max` to `% ` (1) | gap | the tolerance loses its factor `max(\|t\|)` (becomes about `8 eps`), so a valid uniform grid at large time magnitude (e.g. `linspace(1000, 1001, n)` in float64) is refused; all tests use `t` from 0 to at most 1 |
| 827 | `8` to `9` (1) | gap | a grid perturbed by between 8 and 9 times `eps * max` should be refused and is accepted; the existing refusal test perturbs far beyond that. The bound should be measured before pinning |
| 828 | `<=` to `<` (1) | dead | differs only when the deviation equals the tolerance exactly, which for a non-zero grid needs a deviation of exactly `8 eps max`; with `t` identically 0 both sides are 0, a zero step that the solver does not support |
| 828 | `steps[0]` to `steps[1]` (1) | gap | for a two-point `t` there is one step and `steps[1]` raises IndexError; same two-point test as 825 |
| 832 | `y0.shape[1:]` to `[2:]` in the message (1) | equivalent | only the text of the failure message changes; the refusal test matches "must match y0 mode grid" and the message is not the contract |
| 835 | `t[1] - t[0]` to `+` (1) | gap | equal when `t[0] == 0`, and every etdrk4 test starts at 0; a grid starting at `t0 != 0` (e.g. `linspace(1, 2, n)`, linear-decay exact solution) makes `h` wrong |
| 845 | `Q.is_complex()` to `not` (1) | gap | for real eigenvalues the swapped dtype gives equal values, but for complex eigenvalues the weights are cast to real and lose their imaginary part; `etdrk4_solver` is only run with real `linear` (the complex weights are tested through `_etdrk4_weights` only) |

### Gaps

- 51: `placed(torch.device("meta"))` equals `torch.device("meta")` when an accelerator is present.
- 83: `refuse_a_generator_placed_elsewhere` with a stub generator on `cuda:1` and work on `cuda:0` raises.
- 123: `to_complex_coeff(real32, coeff_dtype=torch.complex128).dtype` is complex128.
- 218, 221: fitting a constant column gives `ss == eps` and a finite transform.
- 288, 289: `transform` of an input with a NaN (and an inf) entry returns a finite result (state the intended replacement value first).
- 331: call `StandardScaler.load` on an instance.
- 435: `resize_modes(x, n)` equals `resize_modes(x, n, rescale=True)`.
- 450: `resize_modes` with a target tuple one axis too long and one too short raises AssertionError.
- 483: `resize_modes` to its own shape with 512 (even, above 256) modes returns the input unchanged.
- 614: `zero_coeff` with a band list one entry too long and one too short raises AssertionError.
- 674, 825: `euler_solver` and `etdrk4_solver` with a 0-d, a 2-D, an empty, a one-point and a two-point `t`: the first four raise AssertionError, the two-point case integrates.
- 677: `euler_solver` with an empty `t` raises AssertionError and with two points integrates.
- 687: `euler_solver` with a right-hand side returning NaN raises AssertionError.
- 782: `_etdrk4_weights` on complex eigenvalues equals (`torch.equal`) a reference built from `2 * contour_points` nodes on the full circle.
- 807: `etdrk4_solver` default equals `contour_points=32` exactly.
- 827: a float64 uniform grid at large magnitude (e.g. `linspace(1000, 1001, 50)`) is accepted, and a grid perturbed by between 8 and 9 times `eps * max` (after measuring the true rounding) is refused.
- 828: `etdrk4_solver` with a two-point `t` runs.
- 835: `etdrk4_solver` on `linspace(1, 2, n)` with `N(v) = 0` matches `v0 * exp(L * (t - t0))`.
- 845: `etdrk4_solver` with an imaginary `linear` (advection) and zero nonlinearity matches `v0 * exp(L t)`.
