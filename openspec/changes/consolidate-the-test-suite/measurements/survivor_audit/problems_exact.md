## SpectralSVR/problems/exact.py (45 survivors)

Every consumer of these builders feeds their output into a residual check that is satisfied by any member of the family (a heat-mode or Cole-Hopf field with any wavenumber, amplitude, offset is still an exact solution; a periodic manufactured pair with any `mx`, `mt`, amplitude is still self-consistent), and no test compares a builder's output with a hand-computed value, a shape or a documented range. The heat-mode test runs only at `length = 2*pi`, the manufactured builder only at its defaults.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 35 | `ns + 1` to `ns \| 1`, `ns ^ 1` (2) | gap | equal to `ns + 1` only for even `ns`; for odd `ns` the grid has `ns - 1` (or `ns - 2`) points at the wrong spacing, and tests with odd `modes` (16..32 in the antiderivative test) build the field from the same shortened grid, so it stays self-consistent; no test asserts the grid length or spacing |
| 70 | `max_m + 1` to `- 1`, `* 1`, `// 1`, `** 1`, `\| 1`, `^ 1` (6); `randint(1, ..)` to `randint(2, ..)` (1) | gap | the drawn index range `[1, max_m]` is documented but never checked; tests only use `max_m=3` and accept any index set (e.g. `max_m - 1` draws only 1) |
| 71 | `0.5 + rand` to `-`, `*`, `%`, `**` (4); `0.5` to `1.5` (1) | gap | documented amplitude range `[0.5, 1.5)` is not asserted; any amplitude gives an exact heat-mode field |
| 72 | `2 * pi * rand` to `2 + pi*rand`, `2 - ..`, `2 / ..`, `2 // ..`, `2 % ..`, `2 ** ..` (6); `2` to `3` (1) | gap | documented phase range `[0, 2*pi)` is not asserted; any phase gives an exact field |
| 112 | `2 * pi` to `2 // pi` (1) | gap | gives `k = 0`: `u` is constant in x and `u_x = 0`, so `f = u u_x = 0` and the heat-mode residual test still passes (its tolerance scales with `|u|`); nothing asserts `u_x != 0` or a point value |
| 112 | `/ length` to `// length`, `% length` (2) | gap | the heat-mode test uses `length = 2*pi` only, where `2*pi*m // 2*pi == m` (and `%` also leaves the test blind); a different length (e.g. 1.0 or 3.0) separates them |
| 161 | `amp * rand` to `amp % rand` (1) | gap | documented amplitude range `[0, amp)` not asserted; any amplitude below the theta-positivity cap is still exact |
| 206 | `theta = 1.0 + ...` to `2.0 + ...` (1) | gap | a constant shift of theta is still a heat solution, so `u = -2 nu theta_x / theta` is still an exact Burgers solution; only a hand-computed value, e.g. `u = 2 nu a k sin(kx+b) / (1 + a cos(kx+b))` for one mode, tells them apart |
| 223 | default `mx = 2` to `1` (1) | gap | tests never pass `mx` and the residual checks hold for any integer `mx` on the unit period; the default is documented but never pinned |
| 224 | default `mt = 2` to `3` (1) | gap | as line 223 (nt=16 still resolves mt=3) |
| 225 | default `amp_x = 1.0` to `2.0` (1) | gap | as line 223; any amplitude is a consistent pair |
| 226 | default `amp_t = 0.7` to `1.7` (1) | gap | as line 223 |
| 227 | default `length = 1.0` to `2.0` (1) | gap | as line 223: with `mx = 2`, `length = 2` gives `kx = 2*pi`, one period on `[0, 1)`, still band-limited |
| 228 | default `t_length = 1.0` to `2.0` (1) | gap | as line 227 for `wt` |
| 264 | `x.shape[0]` to `x.shape[-1]` (1) | dead | `x` is documented as 1-D `(ns,)`, where index 0 and -1 agree; a 2-D `x` is unsupported (the `view(1, 1, ns)` would fail) |
| 265 | `2 * pi * mx` to `2 // pi * mx` (1) | gap | gives `kx = 0`: `u_x = u_xx = 0` and `f = u_t` is still self-consistent |
| 265 | `/ length` to `* length`, `** length` (2) | gap | identical at the default `length = 1.0` which every test uses; only a non-unit `length` separates them |
| 265 | `2` to `1` (1) | gap | `kx = pi * mx = 2*pi` at `mx = 2`, still one period on the unit grid, still self-consistent |
| 266 | `2 * pi * mt` to `2 // pi * mt` (1) | gap | gives `wt = 0`: `u_t = 0`, `f` still self-consistent |
| 266 | `/ t_length` to `* t_length`, `** t_length` (2) | gap | identical at the default `t_length = 1.0`; only a non-unit `t_length` separates them |
| 266 | `2` to `3` (1) | gap | `wt = 6*pi` at `mt = 2`: three periods, still band-limited at `nt = 16`, self-consistent |
| 269 | `amp_x * sin(kx X)` to `amp_x + sin(kx X)` (1) | gap | at the default `amp_x = 1.0` this is `u + 1`, a constant shift whose `f` is recomputed from the shifted `u` and the unchanged `u_x`, so it is still an exact pair; needs a point value or `amp_x != 1` |
| 270 | `expand(1, ..)` to `expand(2, ..)` (1) | gap | `f` becomes shape `(2, nt, ns)` while `u` stays `(1, nt, ns)`; the docstring promises `(1, nt, ns)` for both and no test asserts shapes of the builder's output (broadcasting hides it downstream) |
| 271 | `expand(1, ..)` to `expand(2, ..)` (1) | gap | as line 270 |
| 272 | `expand(1, ..)` to `expand(2, ..)` (1) | gap | as line 270 |

Gaps
- Line 35: `space_grid(ns, L)` for odd and even `ns` has shape `(ns,)`, spacing `L / ns`, first node 0, last node `L - L/ns`.
- Lines 70-72, 161: `heat_mode_params` and `cole_hopf_params` outputs lie in the documented ranges (index in `[1, max_m]` and attaining both ends over many draws, amplitude in `[0.5, 1.5)` / `[0, amp)`, phase in `[0, 2*pi)`).
- Line 112: `heat_mode_u_and_ux` against a hand-computed single-mode value at a point, and at `length != 2*pi`; assert `u_x` is nonzero.
- Line 206: `cole_hopf_u` against a hand-computed single-mode value (`theta = 1 + a cos(kx+b)` at `t = 0`).
- Lines 223-228, 264-266, 269: `burgers_periodic_manufactured` at its defaults against hand-computed `u` and `f` at one point, plus a call with `length` and `t_length` != 1 compared with the closed form.
- Lines 270-272: `burgers_periodic_manufactured` returns `u.shape == f.shape == (1, nt, ns)`.
