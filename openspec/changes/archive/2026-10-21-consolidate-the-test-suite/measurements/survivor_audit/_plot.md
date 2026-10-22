## basis/_plot.py (191 survivors)

Plot tests (`tests/basis/test_fourier_basis.py::test_every_plot_of_a_field_draws`, `test_every_spectrum_plot_of_a_field_draws`, the two plot refusals) are `oracle_smoke`: they draw and assert nothing about the artist, legend, extent or data, on 1-function bases with default `i=0, n=1`, no `res`, and no time-dependent basis. Most survivors follow from that.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 16 | `if not TYPE_CHECKING` (1) | equivalent | names under the guard appear only in string annotations; running the imports changes nothing observable (`_plot` is imported lazily after both modules load) |
| 33, 34 | `i` default 0 to 1, `n` default 1 to 2 (2) | dead | `Basis.plot` passes `i`, `n` explicitly; only a direct call of private `plot_basis(basis)` reaches the defaults |
| 37, 39 | `complex_scatter` False to True, `legend` True to False defaults (2) | dead | same: `Basis.plot` always passes both; the public defaults are `__base.py` 810/812 |
| 38 | `\|` to `+ - * / // % ** >> << & ^` in annotation (11) | equivalent | `from __future__ import annotations` on line 7, annotation never evaluated |
| 47 | `i + n` to `i \| n`, `i ^ n` (2) | gap | differ only when `i & n != 0` and the sum overshoots, e.g. `i=1, n=1` on a 1-function basis; the only refusal test is `i=0, n=2` |
| 47 | `<=` to `==`, `is` (2) | gap | differ when `i + n < len(basis)`; every plot test uses a 1-function basis with `i=0, n=1`; `is` equals `==` for small ints |
| 51, 52 | `res is None` flipped (`is not`, `not ... is`), slice step 200 to 201 (3) | gap | no plot test passes `res` and none checks the default 200 points per axis |
| 53 | `ndim + 1` to `- * / // % ** >> << \| & ^`, `+ 2` (12) | gap | no test plots a time-dependent basis; `td.plot()` works on a time-dependent 1-axis basis (checked), so the branch is live |
| 57 | `len(values) > 0` to `!= 0` (1) | equivalent | `len` is never negative |
| 57 | `> 0` to `>= 0`, `> -1`; `values is None` to `is not None` (3) | dead | assertion fails only on an empty selection, reachable only with negative `i` (`i=-1, n=1` raises "something went wrong", checked); `n<=0` selects all (`n=0` plots all, checked) and `i+n>len` is refused on line 47 |
| 65 | `if not basis._complex_funcs` (1) | gap | only legend labels and line count differ (real basis drawn as real+imag); no artist or legend assertion |
| 66 | `if not complex_scatter` (1) | gap | scatter versus line is not asserted (return type, collection count) |
| 70 | `if not legend` (1) | gap | no test checks a legend exists or is absent |
| 71 | `i + j` to `- * ** >> << \| & ^` in the scatter legend label (8) | gap | at `i=0, j=0` all but `**` give label 0 and no legend text is asserted; needs `i>=1, n>=2` and legend text check |
| 82, 97, 119, 129 | same label arithmetic in the real, imaginary, both and real-only legends (8 each) | gap | same: legend text is never read |
| 108 | `plot[0]` to `plot[-1]` (1) | equivalent | `plt.plot(x, y)` returns a one-element list (checked), so `[0]` is `[-1]` |
| 118 | `k == 0` to `!= < <= > >= not`, `== 1` (7) | gap | selects "Real" versus "Imaginary" label; legend text not asserted |
| 120 | `range(2)` to `range(3)` (1) | gap | legend would list 3n labels; matplotlib truncates without error; count not asserted |
| 146 | `grid[0, -1, 1]` to `+1`, `~1`, `1` (3) | gap | changes `x1` and so the image extent and xlim; no extent assertion |
| 146 | `grid[0, 0, 1]` to `grid[1, 0, 1]` (1) | equivalent | `grid[a, b, 1]` depends only on `b` (checked `torch.equal`); other literal siblings of this family were not run |
| 147 | `grid[-1, 0, 0]` to `+1`, `~1`, `1`; `grid[0, 0, 0]` to `grid[1, 0, 0]` (4) | gap | changes `y0` or `y1`; extent not asserted |
| 149, 152 | `shape[k] - 1`, `2 *`, `/`, `x1 - x0` mutated, comparison `> 1` to `== != < <=`, `not`, number 2 to 3 (30 each) | gap | each moves the half-pixel margin of the extent; `im.get_extent()` is never read. Expected extent for an `m`-point grid is `(x0 - h, x1 + h, ...)` with `h = (x1 - x0) / (2 (m - 1))` |
| 149, 152 | `shape > 1` to `>= 1` (1 each) | gap | differs only when that axis has 1 point (ZeroDivisionError); a 2D plot with `res=(slice(0,1,8), slice(0,1,1))` works (checked) |
| 154 | `x0 - half_dx`, `x1 + half_dx`, `y0 - half_dy`, `y1 + half_dy` operators (12) | gap | extent not asserted |
| 162, 164 | `values[0]` to `values[-1]` (2) | gap | identical while `n=1`; 2D plots only the first function; needs `n=2` and an image-data assertion |
| 211 | `i : i + n` to `i \| n`, `i ^ n` (2) | gap | differ for `i&n != 0` (e.g. `i=1, n=1`) and need a multi-function basis; tests use `i=0, n=1` on 1 function |
| 214 | `len(modes) == 1` to `<= 1` (1) | dead | `len(modes)==0` cannot occur: `modes` is `(0,)` with no coefficients and `shape[1:]` of a non-empty field otherwise |
| 214 | `== 1` to `>= 1` (1) | gap | a 2D spectrum would take the 1D branch (survives, so it does not crash); assert the 2D case returns an `AxesImage` |
| 215 | `modes[0]` to `modes[-1]` (1) | equivalent | the branch has exactly one mode axis |
| 221 | `elif len(modes) == 2` to `<= 2` (1) | dead | the `1` case is taken above, `0` cannot occur (see 214) |
| 221 | `== 2` to `>= 2` (1) | gap | a 3-axis Fourier basis (`modes=(4,4,4)`) refuses with `NotImplementedError` (checked) but no test plots one; mutant would hit imshow instead |
| 220 | `i + j` to `- * ** >> << \| & ^` in the spectrum legend (8) | gap | legend text never read; needs `i>=1, n>=2` |
| 223 | `coeff[0]` to `coeff[-1]` (1) | gap | identical while `n=1`; needs `n=2` and a data check |

### Gaps

- 47: `plot(i=1, n=1)` on a 1-function basis raises "out of bounds"; and `plot(n=1)` on a 3-function basis does not.
- 51/52: `plot(res=...)` honours `res`, and the default draws 200 points per axis (read `Line2D.get_xdata()` length).
- 53: `plot()` on a time-dependent basis returns a line artist; a 3-axis time-dependent basis raises `NotImplementedError`.
- 65/66/70/71/82/97/118/119/120/129: on 1D real, complex (both, real, imag, scatter) fields with `i=1, n=2`, assert artist type, line count, and exact legend labels, plus `legend=False` draws none.
- 146/147/149/152/154: for a 2D plot assert `im.get_extent()` against `(x0-h, x1+h, y0-k, y1+k)`, and the 1-point-axis case does not divide by zero.
- 162/164: 2D `plot(n=2)` shows function `i` (compare image data with its values).
- 211/220/223: `plot_coefficients(i=1, n=2)` draws functions 1 and 2, legend labels `Function (1)`, `Function (2)`; the 2D case draws function `i`.
- 214/221: 2D spectrum returns an `AxesImage` (1D branch not taken); a 3D basis raises `NotImplementedError`.
