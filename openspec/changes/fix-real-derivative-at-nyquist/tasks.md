## 1. Measure first

- [ ] 1.1 Record the defect as a measurement script in `measurements/`, run from a clean checkout. For real fields at even and odd mode counts on one and two axes, measure the largest imaginary part relative to the largest real part: on the native grid, off the grid (exact, approximate, time-dependent), on a finer grid, after an odd and an even derivative, and after an integral. Verify it reproduces the proposal's 7.9 (antiderivative `ut` at 16 modes), 8.3% off-grid at 8 modes and 2.6e-8 at 9.
- [ ] 1.2 Copy the archived `benchmark_reads.py` into `measurements/`, extended to every dataset. Record the investigation's DeepONet measurements (base twice, derivative-only prototype once, project antiderivative) as the DeepONet noise floor. Verify the copied probe reproduces the archived hashes on the parent commit.

## 2. Failing first

- [ ] 2.1 Measure the rounding of the closed-form comparisons the tests will make (real reading, odd derivative, integral round trip, complex and native readings) over their ranges in single and double precision, and record each. Set each tolerance a stated factor above its recorded worst case.
- [ ] 2.2 Add the tests naming the five scenarios of "A real field reads its boundary mode as a cosine":
  - `BasisLifecycle` gains an invariant that a real field reads real off its grid after any sequence;
  - fuzzed analytic tests cover the closed-form cosine reading, the odd and even derivatives, and the integral;
  - a fuzzed test pins complex fields to the single-wave closed form, and native grids to `torch.fft.ifft`.

  Pin even and odd counts, one and two axes, and the counts 2 and 4. Verify the real-field tests fail on the parent commit and the complex and native ones pass there.

## 3. Repair

- [ ] 3.1 Give `derivative_eigenvalues` a `complex_funcs` parameter that zeroes the boundary eigenvalue of an odd order on an even count for a real field, and pass the field's flag from `grad` and `integral`. Amend `test_mms_grad_is_spectral_derivative` and `test_mms_integral_inverts_grad` to state `complex_funcs=True` for their arbitrary spectra. Verify the derivative and integral tests of 2.2 pass.
- [ ] 3.2 Read a real field through its odd extension on each even axis in `evaluate`, `evaluate_time_dependent_at`, the approximate path and the inverse transform off the native grid, with the scaling each path needs. Give the static entry points `complex_funcs: bool = False` and pass the field's flag from every instance read. Verify every test of 2.2 passes, and audit every static call in `SpectralSVR/`, `scripts/` and `tests/` that passes coefficients for whether it means a real or a complex spectrum.
- [ ] 3.3 Second pass on structure: verify the extension rule lives in one place and that no path repeats it. Verify that no comment or docstring was added and that prose was removed where the change touched it. Verify that no helper exists only to dodge structuring the code.

## 4. Recorded numbers

- [ ] 4.1 Run the probe of 1.2 in clean worktrees of the parent and the fixed commit, and record which datasets' splits, targets and SpectralSVR predictions move, and by how much. Verify the Burgers datasets come back bit-identical, as the design predicts, or record why not.
- [ ] 4.2 Regenerate `benchmarks/operator_comparison.csv` with `scripts/run_benchmark.py` in a clean worktree of the fixed commit, with free disk checked before and after. Verify the rows of every unmoved dataset are bit-identical to the recorded ones. For each moved row, record its movement against a base rerun next to the DeepONet noise floor of 1.2.

## 5. Verification

- [ ] 5.1 Run cosmic-ray over the changed lines against the tests that cover each, rerun each survivor against the whole suite, and record a verdict for each. Verify every mutant of the changed lines is killed or recorded as equivalent or dead.
- [ ] 5.2 Run `just check` through the resource wrapper. Verify lint is clean and the suite passes, and record the test count and wall time.
