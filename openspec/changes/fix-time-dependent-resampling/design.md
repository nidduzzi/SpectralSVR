## Context

See proposal.md - Why. `to_time_dependent` today asks `get_values` for the field on a grid of `slice(start, stop, nt)` along time, which the periodic sampling scheme reads as the half-open nodes `start + i (stop - start) / nt`, transforms each time row spatially, and reshapes the result to `(len, *modes)`. A time-dependent field is read through `evaluate_time_dependent_at` and the grid evaluation, both of which place slice `i` at `start + i (stop - start) / (nt - 1)` and interpolate linearly between slices.

## Goals / Non-Goals

**Goals:**

- Any slice count of at least two is accepted, and the result has that many slices.
- Slice `i` holds the source's values at the time the representation reads it at.

**Non-Goals:**

- The Burgers manufactured-solution generator's own copy of the sampling, which feeds the benchmark: the next change, so a recorded row moves where it is measured.
- `to_time_independent`, which resamples a time-dependent field onto the periodic grid through the representation's own evaluation; it reads the slices with the convention this change makes `to_time_dependent` write, so it needs no change, and the anchor `test_to_time_independent_values_match_analytic` already pins it.

## Decisions

### Sample at the inclusive slice times explicitly

The conversion evaluates the source at the points `(t_i, x)` with `t_i = start + i (stop - start) / (nt - 1)` and `x` on each spatial axis's periodic grid, rather than asking `get_values` for a time slice, whose periodic scheme can only produce the half-open nodes. The spatial transform of each row and the shape `(len, nt, *modes[1:])` follow from that grid, so the reshape no longer assumes `nt` equals the time axis's mode count.

*Alternative rejected:* keep half-open sampling and change how a time-dependent field is read. Every time-dependent consumer, the benchmark's scoring included, reads slices inclusively, so the reader is the convention and the writer is the defect.

### At least two slices

Inclusive spacing divides by `nt - 1`, so one slice has no spacing. The conversion refuses `nt < 2`, naming the count, rather than producing a field whose single slice could not be placed in time. The default stays the time axis's mode count.

### The anchor pinning half-open placement is rewritten, not loosened

`test_to_time_dependent_slice_spectra_match_analytic` expects slice `j` to hold `g(j / nt) S`. It becomes `g(j / (nt - 1)) S` with the same hand-computed spatial spectrum `S` and the same tolerance; the oracle stays analytic, and the change in expectation is the fix, stated in the commit.

### The between-slice oracle is analytic

Between slices the representation interpolates linearly, so a converted field matches its source there only to interpolation error. The test bounds that by `Δt² / 8 · max |∂²u/∂t²|`, computed from the field's own time spectrum, rather than by a fitted tolerance; at the slice times it requires agreement to float32 rounding, measured first.

## Risks / Trade-offs

- **Something relied on the half-open placement.** → `to_time_dependent` is called by the library only through the Burgers generator's own copy, which this change leaves alone; the suite runs in full before and after, and any test whose expectation moves is named in the commit.
- **Evaluating at explicit points costs more than a transform.** → The conversion is used at dataset build time on small grids; the suite's wall time is recorded before and after.
