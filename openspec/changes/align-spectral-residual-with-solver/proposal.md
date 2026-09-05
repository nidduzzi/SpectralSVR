# Align spectral_residual with the numerical solver

## Why

`Burgers.spectral_residual` and `Burgers._solve_spectral` compute the nonlinear term `u u_x` two different ways. Because the manufactured-solution branch defines its forcing as `f = spectral_residual(u, 0, nu)` and the numerical branch then integrates with the solver's own nonlinear term, the manufactured pair `(u, f)` is not a solution of the equation the solver actually solves. It is self-consistent and wrong: `spectral_residual(u, f, nu)` reads exactly `0.000`, because both sides of that check share the same two defects.

**Defect 1 — the real part.** The solver takes the real part before squaring (`burgers.py:209`); `spectral_residual` multiplies the raw complex values (`burgers.py:292-295`). A Nyquist mode's spectral derivative is purely imaginary on the sample grid — eigenvalue `-i pi N`, values `-i pi N c (-1)^j` — so the solver discards that component and the residual keeps it. The two disagree whenever the field carries Nyquist content, which every even mode count does.

**Defect 2 — de-aliasing.** The solver multiplies the nonlinear term by the Orszag 2/3 mask; `spectral_residual` does not. When the product's bandwidth `2 kmax` overruns the grid, the manufactured forcing absorbs the fold-back.

Measured against the *true continuous residual* on a fine grid, with the source field resized to 16 modes:

| source slots | complex product | real product |
| --- | --- | --- |
| 4 (alias-free, isolates defect 1) | 9.71e-2 | **2.15e-7** |
| 5 (odd, no Nyquist bin — control) | 3.64e-7 | 3.64e-7 |
| 8 | 1.05e-1 | 3.35e-2 |
| 16 | 2.60e-1 | 2.59e-1 |

A third measurement, on a field whose product is fully resolved so every variant should agree, isolates a defect in a fourth implementation:

| implementation | error vs analytic | ratio |
| --- | --- | --- |
| solver (2/3 mask, flux form, real part) | 2.1e-8 | 1.0000 |
| `spectral_residual` (no de-aliasing, advective form) | 2.3e-8 | 1.0000 |
| `Burgers.rhs` (3/2 padding, flux form) | 3.3e-1 | **0.6667** |

The middle row is the control: `spectral_residual` is correct here precisely because this field has no Nyquist content and no bandwidth overrun, which is independent confirmation of the isolation above.

Defect 1 is isolated where no aliasing can exist. Defect 2 is what survives its fix, and its size tracks how far the product overruns the grid. The odd-mode-count row is a control that has neither defect and reads clean under both variants.

## What Changes

- `spectral_residual` forms the nonlinear term the way the solver does: take the real part of the sampled values before multiplying, and apply the same de-aliasing mask.
- `Burgers.rhs` is deleted. It is a fourth implementation of the same term and is wrong by exactly `2/3`: it de-aliases by 3/2 padding through `resize_modes(..., rescale=False)`, which scales the function by `1/1.5` on the way up, `1/2.25` once squared, and `x1.5` on the way back — a net `0.6667`, matching the measured ratio. Its only caller is its own smoke test, which asserts shape and absence of NaN and so cannot see a factor of 1.5. Its stated purpose, "kept for future numerical solvers", is served by `_solve_spectral` with a supplied solver since `unify-solver-protocol`. Deleting it also removes the second de-aliasing strategy from the file, leaving one.
- `mms_solution` band-limits to `m//3` slots instead of `m//2`. Once both sides de-alias, a pair whose product overruns the 2/3 band cannot satisfy the continuous equation at all, because the mask discards real content. The threshold is exact: the product reaches `2*kmax` with `kmax ~ half/2`, and the band keeps `m/3`, so `half <= m/3` is the boundary rather than a margin. Measured at `m = 16`, where the band keeps `|k| <= 5`:

| slots | product reaches | fits band | continuous residual |
| --- | --- | --- | --- |
| 8 (`m//2`, current) | 8 | no | 1.50e-1 |
| 7 (`m//2 - 1`) | 6 | no | 7.56e-2 |
| 5 (`m//3`) | 4 | **yes** | **3.64e-7** |
| 4 | 4 | yes | 2.32e-7 |

- A new anchor checks a manufactured pair against the **true continuous residual**, evaluated off the sample grid. The existing check evaluates only on-grid, where a wrong pair still reads `0.000`, so it cannot fail for this class of defect.
- The capability states that a manufactured pair must satisfy the continuous equation, not merely its own discrete residual.
- The Burgers MMS `f -> u` benchmark row is regenerated.

**Explicitly not in scope.** The Nyquist *representation* is correct and is not touched: a lone Nyquist coefficient evaluates to the right cosine (`1.3e-7` against `scale c cos(pi N x)`) and its derivative is right (ratio `1.0000`). The imaginary part it carries off-grid is discardable, so `imag/real` is not a defect metric. `enforce_hermitian`, `generate_coeff` and `resize_modes` are all correct and unchanged.

## Capabilities

### New Capabilities
- `manufactured-solutions`: what a manufactured `(u, f)` pair guarantees — that it satisfies the continuous equation, that its nonlinear term is formed identically to the solver's, and how that is verified.

### Modified Capabilities
- `numerical-anchor-tests`: adds the off-grid continuous-residual anchor, and records that an on-grid self-consistency check cannot serve as an oracle for the code that produced it.

## Impact

- `SpectralSVR/problems/burgers.py`: `spectral_residual`, and the removal of `Burgers.rhs`.
- `tests/problems/test_problems.py`: removal of `test_burgers_rhs_smoke` along with the function it covers.
- `tests/problems/test_problems.py`: the continuous-residual anchor plus the odd-mode-count control.
- `results/bench_results.csv`: the Burgers MMS row changes. Both `u` and `f` move, not only `f`, because the band-limit narrows the manufactured solutions — so the `f -> u` problem becomes genuinely easier, a change in the dataset's difficulty and not merely its numbers. That must be stated where the row is reported. `generate_coeff` is untouched, so the antiderivative, numerical-Burgers and DeepONet-GRF rows are unaffected and seeded numerical Burgers output stays byte-identical.
- Unblocks the round-trip anchor in `time-dependent-forcing-in-numerical-burgers`, which must not be claimed here — it additionally needs `Burgers.generate` to accept a Basis forcing, which is that change's own work.
