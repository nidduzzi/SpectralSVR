## Context

See proposal.md for the defect and its measurement. `Burgers.generate` dispatches on its arguments: with `u0="random"` and `f="random"` it builds a manufactured pair through `mms_solution`, a 2-D field periodic in time and space. Asked for a time-dependent pair, it then converts both fields with a local helper that samples them through `get_values(res=(time_domain, ...))`. `get_values` samples on the field's own scheme, periodic and half-open in time, while the time-dependent representation reads its slices at inclusive times. `Basis.to_time_dependent(nt)` was fixed in `fix-time-dependent-resampling` to sample at exactly those inclusive times, and is held to it by `coefficient-encodings`.

## Goals / Non-Goals

**Goals:**
- Each slice of a time-dependent manufactured pair holds the field and forcing at its stated time.
- No recorded number moves.

**Non-Goals:**
- The numerical branch. It integrates on `t = linspace(start, stop, nt)` and stores the solution at those times, which already agrees with the representation.
- The non-time-dependent manufactured pair, which is periodic in time and correctly sampled on its own grid.

## Decisions

**Use the library's conversion, not a corrected local helper.** The manufactured branch calls `to_time_dependent(nt=...)` on both fields and drops its local helper. A corrected helper would repeat the conversion `coefficient-encodings` already specifies and tests, and would drift from it again the next time either changes. Against the field's closed-form Fourier series, the replacement is off by 1.5e-7 to 4.5e-7 of the largest value at 2, 5, 11 and 50 slices. Before the fix the misses were 1.80, 0.86, 0.49 and 0.11.

**Oracle.** The expected value of each slice is the manufactured field's Fourier series, summed explicitly in the test at the slice times `linspace(start, stop, nt)` and the periodic spatial nodes. The sum uses the coefficients of the same pair generated without time dependence and no basis evaluation, so the test is `oracle_analytic`. The first version compared against the library's own point evaluation. That shared the library's Fourier convention, and beyond a dense-work budget it switches to an approximate NUFFT, wrong by about 1e-4 at (101, 48) modes.

**Ranges.** Modes run from 6 on each axis. `widest_manufactured_band` is `modes // 6`, so at 4 or 5 modes the field is constant along that axis and a sampling defect on it is invisible. Slices run from 2 to 40. Time and space domains are fuzzed, starts from -3 to 3 and lengths from 0.5 to 3, because a unit domain cannot see a conversion that hard-codes the interval. The difference is divided by the expected field's own largest value. The first version divided by the larger of that value and one, which for a field peaking at 0.03 to 0.4 made the bound absolute and about 300 times looser than the rounding.

**What the test cannot see.** The manufactured field is periodic in time, so its value at the interval's end equals its value at the start. A conversion that wrapped the last slice to the start would pass at that slice. Inclusive and half-open sampling differ only at the interior slices, and the test checks those. For the same reason, reversing the slice order is equivalent at 2 and 3 slices. `generate` stores coefficients in single precision and accepts no precision argument, so the test checks single precision only. The double-precision rounding of the conversion is measured at the conversion level.

**Absorbed tests.** `test_burgers_manufactured_branch_shapes` asserted that the pair is time-dependent, has `n` functions and is finite. `test_burgers_generate_honors_domain_start` asserted that the domain's starts reach the field. The new test asserts both on every draw, so both are removed. A state machine does not fit: the requirement constrains one call, and no sequence of calls reaches a state a single call cannot.

**Slice positions are one value type.** Five read paths place a time between slices. `get_values` computed the index in single precision and did not bound it. `to_time_independent` and the time-dependent `evaluate` bounded it through `fractional_slice_index`. `evaluate_time_dependent_at` bounded it and then repeated the interpolation by hand, with a `nan_to_num` to hide the division by zero at a whole index. Resizing in time built its own index. `SlicePositions` in `utils` replaces all of these. It is a frozen dataclass built only from times, an interval and a slice count. It computes in double precision, bounds the result to the slices, and refuses a non-finite time, an empty interval or no slices. It holds the lower slice, the upper slice and the weight, so a consumer never forms an index itself. `interpolate_tensor` takes positions instead of a raw index, so an index past the last slice cannot be passed. `fractional_slice_index` and the `interpolate_time_tensor` wrapper are removed. A time outside the interval still reads at the nearer end where it did before. `evaluate_time_dependent_at` still refuses such a time before forming positions.

**Recorded numbers.** The benchmark's manufactured case passes `time_dependent_coeff=False`. The forced and numerical cases take the numerical branch. Every Burgers generator call in the committed notebooks passes `time_dependent_coeff=False`, and the notebook calls on the default path are `FourierBasis.generate`, not the Burgers generator. The manufactured change is therefore unreachable from any recorded number. The slice-position change is not: every time-dependent read now computes its weights in double precision, which can move a recorded number at the rounding level. The forced and numerical benchmark cases are rerun from a worktree of the parent commit with its own environment and compared against the recorded noise floor. The diff argument is stated in the tasks, and grepping `scripts/` and `notebooks/` for the time-dependent manufactured call checks it.

**Resource bounds on every run.** Every test, measurement and mutation run of this change goes through one wrapper. It is a systemd user scope with:
- 16 GiB memory and no swap;
- a 1200% CPU quota, which is 12 of the 20 cores;
- 1024 tasks and IO weight 50;
- nice 10 and the idle IO class;
- 2 threads per process;
- the first GPU only;
- 4096 open files, no core dumps, and at most 8 GiB per written file;
- a dedicated temporary directory.

Each limit was checked from inside the scope. Accelerator memory is capped by the process itself: `tests/conftest.py` gives pytest a quarter of the device split across its workers, and `scripts/run_benchmark.py` sets the same quarter.

## Risks / Trade-offs

- [A cgroup cannot cap total disk use, and GPU compute cannot be capped below a whole device without CUDA MPS] → the per-file size limit, the dedicated temporary directory and a free-space check before and after each heavy run bound storage. GPU use is bounded by allowing one device and capping its memory, not its compute share. Both limits are stated rather than implied.
- [A caller relied on the half-open slices] → the four existing call sites check shapes, devices and that a residual runs, not slice values. The change is still a behaviour change for any external caller of the default path, and the commit says so.
