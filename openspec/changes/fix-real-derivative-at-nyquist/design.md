## Context

See proposal.md for the defect and the investigation. Four code paths read the boundary mode of a stored spectrum:

- `_dense_evaluate`, which builds one exponential factor per axis from `wave_number`;
- `nufft_evaluate`, which reorders the spectrum into ascending wavenumbers for torchkbnufft;
- `fn`, the explicit basis matrix behind the time-dependent `evaluate`, `evaluate_time_dependent_at`, and the inverse transform at any resolution other than the mode count;
- `derivative_eigenvalues`, behind `grad`, `integral` and the Burgers solver.

The inverse transform at the native resolution uses the FFT, which samples only the nodes. There the single wave and the cosine agree.

`resize_modes` already reads the boundary mode as shared between `+N/2` and `-N/2`. Enlarging an even grid by one splits its boundary coefficient evenly, and reducing it back recombines the halves exactly. `coefficient-encodings` holds this under "Resizing conserves content at the boundary mode".

## Goals / Non-Goals

**Goals:**
- A real field reads real everywhere, and its odd derivatives and its integrals drop the boundary mode, at every mode count.
- A complex field, and any field on its own grid, reads exactly as before.
- Every recorded number that moves is measured, regenerated from a clean commit and compared against its base.

**Non-Goals:**
- The time axis of a time-dependent field, which is not a Fourier axis.
- The forward transform, which already stores a real field's boundary coefficient as the real sum the cosine reading needs.
- A tuned or fair DeepONet comparison. This change records DeepONet's sensitivity, and `fair-benchmark-plan` uses it.

## Decisions

**Read the cosine through the odd extension the resize already defines.** On each even axis, a real field is read through its spectrum enlarged by one mode with `resize_modes`. That splits the boundary coefficient evenly between `+N/2` and `-N/2`, and every reading path then sums an odd spectrum, where the boundary is unambiguous. Writing a cosine column into each of `fn`, `_dense_evaluate` and `nufft_evaluate` was the alternative. It would put the same rule in three places that would have to stay in step, two of which build their factors differently. The extension states it once, through a resize the specs already hold.

Scaling follows the path:
- A path that divides by the spectrum's own size (`evaluate`, `evaluate_time_dependent_at`, `nufft_evaluate`) reads the extension with `rescale=True`. That multiplies by `(N + 1) / N` per extended axis, so dividing by the enlarged size leaves the stored size's normalisation.
- A path that returns an unscaled sum for its caller to divide by the stored size (the inverse transform off the native grid) reads it with `rescale=False`.

A single test holds both against one closed form, so neither can drift.

**Drop the boundary mode in odd-order differentiation.** For a real field, `derivative_eigenvalues` returns zero at the boundary mode for an odd order on an even count. That is the coefficient-space form of enlarging, differentiating and reducing: the halves at `+N/2` and `-N/2` pick up opposite eigenvalues and cancel. `integral` already zeroes any mode whose eigenvalue is zero, as it does the mean, so it drops the boundary mode with no further code. An even order keeps the shared eigenvalue `(i pi N / L)^ord`, which is the same at both mode numbers.

**`complex_funcs` decides, on the field and on every static entry point.** The field's flag already means "this field is real" when False, and `generate_coeff` already enforces conjugate symmetry under it. Instance reads pass `self.complex_funcs`. The static `evaluate`, `inv_transform`, `fn`, `evaluate_time_dependent_at` and `derivative_eigenvalues` take `complex_funcs: bool = False`, matching the field's default. So a static caller who means a complex spectrum says so, as the two amended tests do. Inspecting coefficients for symmetry at run time was rejected: it is a data-dependent branch with a threshold, and the type already states the claim.

**The native grid is not touched.** The FFT path samples only the nodes, where `cos(pi j) = (-1)^j` is exactly what the single wave gives. It stays as it is, and a test pins bit-identity there.

**Oracles.**
- **Real reading:** a closed-form sum in the test, with the boundary term written as `c cos(pi N (x - a) / L)` and every other term as its exponential. No basis call, so `oracle_analytic`.
- **Odd derivative:** the closed form differentiated by hand.
- **Complex and native readings:** the single-wave closed form and `torch.fft.ifft` respectively, both independent of the changed code.
- **"Reads real after any sequence":** this holds across a sequence of operations (generate, resize, differentiate, integrate, convert through time), so it becomes an invariant of `BasisLifecycle`, not a new single-shot test. Tolerances are measured first and recorded.

**Recorded numbers.**
1. The archived change's benchmark-reads probe is copied into this change. It hashes split, query, targets and SpectralSVR predictions for every dataset and seed, before and after.
2. The datasets that move are regenerated by `scripts/run_benchmark.py` in a clean worktree of the fixed commit.
3. The whole recorded table is rerun there. Rows of unmoved datasets must come back bit-identical, which the archived change showed this harness does on CUDA.
4. The investigation's measurement of DeepONet under the derivative-only prototype is recorded as the noise floor for DeepONet rows: up to 0.057 rrse per seed while unrepaired reruns are bit-identical.

**Burgers.** The solver multiplies `d1` by a de-aliasing mask that keeps `|k| <= (2/3)(N/2)`, so the boundary eigenvalue never reaches it. The manufactured band is at most `N // 6`. Burgers is expected not to move, and the hashes check it.

**Resource bounds.** Every run goes through `/tmp/claude-1000/bounded.sh`: memory, CPU quota, tasks, IO, threads, one GPU with the allocator at a quarter, file size, open files, temporary directory. Free disk is checked before and after the benchmark runs.

## Risks / Trade-offs

- [A static caller passing a complex spectrum without `complex_funcs=True` now gets the cosine reading] → the library's own static calls are audited in the tasks. The default matches the field's default, so a caller relying on the old reading is already relying on a real field being read as complex.
- [Enlarging allocates one more mode per even axis on every off-grid read] → the cost is one mode per axis. The native-grid FFT path, the common one, pays nothing.
- [Regenerating the table moves DeepONet rows by more than the repair's own effect] → that is the finding the noise floor records. The regenerated rows carry the fixed commit in their provenance, and the commit message states the movement per dataset against the base rerun.
