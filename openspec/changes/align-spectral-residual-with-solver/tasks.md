## 1. One nonlinear term, two callers

- [x] 1.1 Extract the spectral nonlinear term `-0.5 * d1 * transform(real(u)^2) * dealias` into a helper on `Burgers` taking the field's coefficients, the first-derivative eigenvalues and the de-aliasing mask; have `_solve_spectral` call it; verify the full suite passes and the seeded numerical Burgers field (`nu=0.01`, `(16,16)`, `slice(0,1,50)`, seed 1234) is `torch.equal` to a worktree at this branch's parent commit
- [x] 1.2 Route `spectral_residual` through the same helper, taking the real part before multiplying and applying `basis.dealias_mask` at the field's own mode count; verify a basis returning `None` for the mask behaves exactly as before on both call sites
- [x] 1.3 Verify the existing MMS tests that assert a zero discrete residual still pass, since both sides of that check move together

## 1b. Remove the fourth implementation

- [x] 1b.1 Delete `Burgers.rhs` (`burgers.py:255-270`) and `test_burgers_rhs_smoke` (`tests/problems/test_problems.py:556`); verify `grep -rn "Burgers.rhs\\|\\.rhs(" SpectralSVR/ tests/ scripts/ notebooks/` returns nothing but unrelated matches
- [x] 1b.2 Verify the only remaining de-aliasing strategy in `burgers.py` is the 2/3 mask, so `resize_modes` is no longer used for de-aliasing anywhere in the file
- [x] 1b.3 Verify `tests/test_oracle_coverage.py` still passes after removing an `oracle_smoke` test, and that the module retains at least one anchor

## 1c. Band-limit the manufactured solution

- [x] 1c.1 Change `mms_solution`'s band-limit from `m//2` to `m//3` slots; verify the generated field's highest mode satisfies `2 * kmax <= m // 3` at mode counts 8, 16 and 32, so the product fits the de-aliasing band by construction
- [x] 1c.2 Verify the continuous residual drops to rounding level at each of those mode counts, and record the measured values; without this the alignment in group 1 is necessary but not sufficient
- [x] 1c.3 Verify the existing MMS tests still pass and that `mms_solution`'s docstring states the band-limit is set by the de-aliasing band rather than chosen arbitrarily

## 2. The off-grid anchor

- [x] 2.1 Add a helper in the problem tests that evaluates the continuous Burgers residual for a `(u, f)` pair on a grid finer than and offset from the field's sample points, using the field's own exact spectral derivatives; verify it returns a relative figure, not an absolute one
- [x] 2.2 Add `test_manufactured_pair_satisfies_the_continuous_equation` (`oracle_analytic`, `no_fuzz`): assert the continuous residual is at rounding level for mode counts 8, 16 and 32; record the measured values in the test; verify it FAILS with the pre-change convention (measured 9.71e-2 at 4 source slots, 2.60e-1 at 16) and passes after
- [ ] 2.3 Add the odd-mode-count natural control to the same test: a field built with an odd mode count has no unpaired top mode and must read clean under either convention; verify it passes both before and after the fix, which is what makes it a control rather than a second assertion
- [x] 2.4 Add the negative control: perturb a known-good manufactured forcing by one percent and assert the residual rises far above its clean value; record both numbers; verify the separation is at least three orders of magnitude
- [x] 2.5 Verify `tests/test_oracle_coverage.py` passes and the new anchor carries `oracle_analytic`, not `oracle_consistency` — its oracle is the continuous equation, not the code's own output

## 3. Benchmark regeneration

- [x] 3.1 Regenerate `results/bench_results.csv` with `scripts/run_benchmark.py`; verify the Burgers MMS `f -> u` row changes and record both the old and new values in the change's design or a results note so the discontinuity is visible. Record that both `u` and `f` moved, so the dataset's difficulty changed and not only its numbers
- [x] 3.2 Verify the antiderivative, numerical-Burgers and DeepONet-GRF rows are unchanged, which also evidences that `generate_coeff` was untouched

## 4. Verification

- [x] 4.1 Run `just check`; verify clean
- [x] 4.2 Verify no source file gained an explanatory prose comment; the shared helper's name must carry the intent, with any numerical rationale confined to a docstring
- [x] 4.3 Verify `Antiderivative.spectral_residual` either shares the convention or is documented as having no nonlinear term, so the open question in the design is closed rather than left dangling
- [x] 4.4 Verify the finite-difference `residual` at `burgers.py:328` is untouched and still shares no code with the spectral path, so it remains an independent cross-check rather than becoming a second view of the same computation
