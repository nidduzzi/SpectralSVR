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

**Use the library's conversion, not a corrected local helper.** The manufactured branch calls `to_time_dependent(nt=...)` on both fields and drops its local helper. A corrected helper would repeat the conversion `coefficient-encodings` already specifies and tests, and would drift from it again the next time either changes. The probe that measured the defect also measured this replacement: 1.7e-7 to 1.9e-7 relative to the largest value at 5, 11 and 50 slices, against 0.86, 0.49 and 0.11 before.

**Oracle.** The new test generates the same pair twice from the same seed, once time-dependent and once not, and evaluates the non-time-dependent pair at the inclusive slice times. The non-time-dependent pair is the manufactured field itself, which the continuous-residual anchors already hold to the equation, and its evaluation shares no conversion code with the branch under test. Because the expected values are the library's own output, the test is marked `oracle_consistency`. It is fuzzed over slice counts and mode counts, with these edges pinned: 2 slices, slices equal to the time mode count, fewer and more slices than time modes, and odd and even counts. The tolerance is measured first over those ranges in single and double precision, recorded, and set a stated factor above the recorded worst case.

**Recorded numbers.** The benchmark's manufactured case passes `time_dependent_coeff=False`. The forced and numerical cases take the numerical branch. Every Burgers generator call in the committed notebooks passes `time_dependent_coeff=False`, and the notebook calls on the default path are `FourierBasis.generate`, not the Burgers generator. The changed code is therefore unreachable from any recorded number. The diff argument is stated in the tasks, and grepping `scripts/` and `notebooks/` for the time-dependent manufactured call checks it.

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
