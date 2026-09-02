## 1. Stage times (Bug 1)

- [ ] 1.1 In `etdrk4_solver`, evaluate the four nonlinear stages at `t[i-1]`, `(t[i-1]+t[i])/2`, `(t[i-1]+t[i])/2`, `t[i]` instead of `ti` for all four; verify the full existing suite still passes (nothing in it depends on `t`)
- [ ] 1.2 Measure the final relative error of `test_etdrk4_mms_time_dependent_forcing` at its 300 steps with the shipped all-stages-at-`t_n` loop and with the corrected loop, and record both numbers in the test body; verify the two differ by at least two orders of magnitude
- [ ] 1.3 Tighten that test's tolerance to one order of magnitude above the corrected error; verify it fails when the stage times are temporarily reverted to `ti` and passes on the corrected code

## 2. Uniform grid

- [ ] 2.1 Assert `t.diff()` is uniform (`torch.allclose` against its first element) next to the existing `t.ndim`/`len(t)` assertions in `etdrk4_solver`; verify a `linspace` grid passes and a concatenated two-spacing grid raises `AssertionError` before any step

## 3. Contour weights (Bug 2)

- [ ] 3.1 Extract the weight computation into a named helper that takes `linear`, `h`, `contour_points` and returns `Q, f1, f2, f3`, with the real-eigenvalue branch byte-for-byte the current half-circle `.real` code; verify `git diff` of the moved lines shows only relocation, and the full suite passes
- [ ] 3.2 Add the complex branch: when `torch.is_complex(linear) and bool(linear.imag.any())`, use `2 * contour_points` nodes on the full unit circle and keep the complex mean; verify the branch is selected for `linear = torch.tensor([-2j])` and not for `FourierBasis.derivative_eigenvalues(16, 1.0, ord=2) * 0.01`
- [ ] 3.3 Cast the weights to the state's complex dtype on the complex branch and to its real dtype on the real branch, as today; verify `etdrk4_solver` runs to completion on a purely imaginary `linear` with `N = 0` and its output matches `y0 * exp(L t)` at the final time to `1e-10`

## 4. Documentation

- [ ] 4.1 Update the `etdrk4_solver` docstring to state that eigenvalues may be complex, that the time grid must be uniform, and that the real-eigenvalue path is the half-circle shortcut; verify the docstring names all three
- [ ] 4.2 Update the `Burgers._solve_spectral` docstring so "any basis whose differentiation is diagonal" is qualified by the uniform-grid requirement and no longer implies untested operator classes; verify the text no longer over-claims

## 5. Tests

- [ ] 5.1 Add `test_etdrk4_order_with_time_dependent_forcing` (`oracle_invariant`, `no_fuzz`): scalar `v' = -200 v + N(t)` with exact `sin(3t) + 2`, steps `40, 80, 160, 320, 640, 1280`, asserting measured order `> 3.9` on the last three halvings and recording the coarser orders without asserting; verify it passes on the corrected code and fails (order near 1) with stage times reverted
- [ ] 5.2 Add `test_etdrk4_complex_weights_match_closed_form` (`oracle_analytic`, `no_fuzz`): `L = {-2i, -20i, -200i, 5i}`, `h = 0.1`, each weight vs the closed-form expression to `1e-10`; verify it passes on the corrected code and fails (errors ~1e-2) with the half-circle `.real` path forced
- [ ] 5.3 Add `test_etdrk4_real_weights_unchanged` (`oracle_consistency`, `no_fuzz`): weights for `linear = -0.01 * k**2` stored as `complex64` with zero imaginary part are `torch.equal` to a literal half-circle `.real` computation written inline in the test; verify it passes
- [ ] 5.4 Add `test_etdrk4_rejects_non_uniform_grid` (`oracle_smoke`, `no_fuzz`): `torch.cat([linspace(0, 0.5, 6), linspace(0.6, 1.0, 3)])` raises `AssertionError`; verify it passes
- [ ] 5.5 Verify `tests/test_oracle_coverage.py` still passes with the new markers, and that no new test relies solely on the solver's own output as its oracle

## 6. Verification

- [ ] 6.1 Byte-identity: generate a seeded numerical Burgers field (`nu=0.01`, `(16,16)`, `slice(0,1,50)`, seed 1234) on this branch and in a `git worktree` at the parent commit; verify `torch.equal` on both `u` and `f` coefficients
- [ ] 6.2 Run `just check` (ruff, mypy, ty on `SpectralSVR scripts`, full pytest); verify clean
- [ ] 6.3 Verify no source file gained an explanatory prose comment; the branch and the stage times must read from names and structure, with the contour rationale confined to the docstring
