## 1. One nonlinear term, two callers

- [ ] 1.1 Extract the spectral nonlinear term `-0.5 * d1 * transform(real(u)^2) * dealias` into a helper on `Burgers` taking the field's coefficients, the first-derivative eigenvalues and the de-aliasing mask; have `_solve_spectral` call it; verify the full suite passes and the seeded numerical Burgers field (`nu=0.01`, `(16,16)`, `slice(0,1,50)`, seed 1234) is `torch.equal` to a worktree at this branch's parent commit
- [ ] 1.2 Route `spectral_residual` through the same helper, taking the real part before multiplying and applying `basis.dealias_mask` at the field's own mode count; verify a basis returning `None` for the mask behaves exactly as before on both call sites
- [ ] 1.3 Verify the existing MMS tests that assert a zero discrete residual still pass, since both sides of that check move together

## 2. The off-grid anchor

- [ ] 2.1 Add a helper in the problem tests that evaluates the continuous Burgers residual for a `(u, f)` pair on a grid finer than and offset from the field's sample points, using the field's own exact spectral derivatives; verify it returns a relative figure, not an absolute one
- [ ] 2.2 Add `test_manufactured_pair_satisfies_the_continuous_equation` (`oracle_analytic`, `no_fuzz`): assert the continuous residual is at rounding level for mode counts 8, 16 and 32; record the measured values in the test; verify it FAILS with the pre-change convention (measured 9.71e-2 at 4 source slots, 2.60e-1 at 16) and passes after
- [ ] 2.3 Add the odd-mode-count natural control to the same test: a field built with an odd mode count has no unpaired top mode and must read clean under either convention; verify it passes both before and after the fix, which is what makes it a control rather than a second assertion
- [ ] 2.4 Add the negative control: perturb a known-good manufactured forcing by one percent and assert the residual rises far above its clean value; record both numbers; verify the separation is at least three orders of magnitude
- [ ] 2.5 Verify `tests/test_oracle_coverage.py` passes and the new anchor carries `oracle_analytic`, not `oracle_consistency` — its oracle is the continuous equation, not the code's own output

## 3. Benchmark regeneration

- [ ] 3.1 Regenerate `results/bench_results.csv` with `scripts/run_benchmark.py`; verify the Burgers MMS `f -> u` row changes and record both the old and new values in the change's design or a results note so the discontinuity is visible
- [ ] 3.2 Verify the antiderivative, numerical-Burgers and DeepONet-GRF rows are unchanged, which also evidences that `generate_coeff` was untouched

## 4. Verification

- [ ] 4.1 Run `just check`; verify clean
- [ ] 4.2 Verify no source file gained an explanatory prose comment; the shared helper's name must carry the intent, with any numerical rationale confined to a docstring
- [ ] 4.3 Verify `Antiderivative.spectral_residual` either shares the convention or is documented as having no nonlinear term, so the open question in the design is closed rather than left dangling
