## Context

See proposal.md - Why. `to_time_dependent` today asks `get_values` for the field on a grid of `slice(start, stop, nt)` along time, which the periodic sampling scheme reads as the half-open nodes `start + i (stop - start) / nt`, transforms each time row spatially, and reshapes the result to `(len, *modes)`. A time-dependent field is read through `evaluate_time_dependent_at` and the grid evaluation, both of which place slice `i` at `start + i (stop - start) / (nt - 1)` and interpolate linearly between slices.

## Goals / Non-Goals

**Goals:**

- Any slice count of at least two is accepted, and the result has that many slices.
- Slice `i` holds the source's values at the time the representation reads it at.

**Non-Goals:**

- The Burgers manufactured-solution generator's own copy of the sampling, which feeds the benchmark: the next change, so a recorded row moves where it is measured.
- ~~`to_time_independent` needs no change.~~ *Corrected after the gate's first run:* it reads slice `j` at the inclusive time but transforms it as the periodic sample at `j T / N`, the mirror of the defect fixed here, and its anchor encodes that. It joins this change; see the decision below.

## Decisions

### Sample at the inclusive slice times explicitly

The conversion evaluates the source at the points `(t_i, x)` with `t_i = start + i (stop - start) / (nt - 1)` and `x` on each spatial axis's periodic grid, rather than asking `get_values` for a time slice, whose periodic scheme can only produce the half-open nodes. The spatial transform of each row and the shape `(len, nt, *modes[1:])` follow from that grid, so the reshape no longer assumes `nt` equals the time axis's mode count.

*Amended in implementation:* the points are not evaluated one by one. The source's inverse transform takes a sampling scheme per axis, so the conversion asks it for `ClosedUniform` nodes along time and the basis's own scheme along each spatial axis. That is the same grid, and the spatial transform is then given the same spatial schemes.

*Alternative rejected:* keep half-open sampling and change how a time-dependent field is read. Every time-dependent consumer, the benchmark's scoring included, reads slices inclusively, so the reader is the convention and the writer is the defect.

### At least two slices

Inclusive spacing divides by `nt - 1`, so one slice has no spacing. The conversion refuses `nt < 2`, naming the count, rather than producing a field whose single slice could not be placed in time. The default stays the time axis's mode count.

### The anchor pinning half-open placement is rewritten, not loosened

`test_to_time_dependent_slice_spectra_match_analytic` expects slice `j` to hold `g(j / nt) S`. It becomes `g(j / (nt - 1)) S` with the same hand-computed spatial spectrum `S` and the same tolerance; the oracle stays analytic, and the change in expectation is the fix, stated in the commit.

### The between-slice oracle is analytic

Between slices the representation interpolates linearly, so a converted field matches its source there only to interpolation error. The test bounds that by `Δt² / 8 · max |∂²u/∂t²|`, computed from the field's own time spectrum, rather than by a fitted tolerance; at the slice times it requires agreement to rounding, measured first. *Amended:* the test builds its fields in double precision, so the coefficients are complex128; the worst slice-time error over 5000 draws was 8.2e-15, and the between-slice error reached 0.992 of the bound.

### The inverse reads the slices at the periodic times by interpolation

*Decided by the user after the gate's first run.* `to_time_independent` evaluates the time-dependent field at the periodic times `start + i (stop - start) / N`, through the same linear interpolation every reader of a time-dependent field uses, and transforms the result. The time axis keeps `N` modes. A round trip of a field periodic in time then holds within `Δt² / 8 · max |∂²u/∂t²|` rather than exactly: the exactness before was the two misplacements cancelling.

*Alternative rejected:* treat slices `0 .. N-2` as the periodic grid and drop the endpoint, which duplicates the start for a periodic field. The round trip would be exact for periodic sources, but the time axis would have `N - 1` modes, which changes the shape contract, and a field that is not periodic in time, such as a Burgers evolution, would lose its last slice.

## Decided after the review

*Every finding in scope, decided by the user.*

**The result is constructed, not copied and mutated.** Each conversion builds its result with the class's constructor, passing coefficients, `time_dependent`, domain, strategy and a sampling with the right number of axes: the time scheme dropped going to time-dependent and prepended coming back. The copy-then-assign form carried the source's per-axis sampling onto a field with one axis fewer, so the inverse applied the time scheme to space; with `(ClosedUniform(), PeriodicUniform())` the round trip differed from the default-sampling one by 5.34. Construction also removes the order dependence between assigning `time_dependent` and `coeff`.

**The per-axis schemes are a property.** ~~`spatial_sampling`~~ `axis_sampling` (*renamed in implementation:* on a time-independent field it covers every axis, the time axis included) resolves one scheme per mode axis, excluding a time-dependent field's time axis; both conversions and `get_values` read it instead of each resolving `samplings_to_tuple` themselves. The forward conversion takes its spatial schemes as `axis_sampling[1:]`.

**The time axis is named.** The inverse samples time on `PeriodicUniform()`, as the requirement states, rather than on `default_sampling()`, which agreed only because `FourierBasis` is the only subclass.

**A time-dependent field converts only to its own slice count.** With no slice count or its own, `to_time_dependent` returns a copy; with another it raises naming both. Resampling between slice counts was rejected as new behaviour with its own bound.

**Refusals are named and stated.** A non-integer or boolean slice count raises `TypeError`; a count below two, an empty basis and a field without a spatial axis raise `ValueError`, each saying which. The contracts live in one-line numpy docstrings with `Raises` sections, not in prose.

**The inverse test's oracle is independent through an anchored pair.** It sets up and reads back through `FourierBasis.transform` and `inv_transform`, whose scale and layout other anchors pin; its expectation, `numpy.interp`, is independent of the code under test.

**Tolerances are measured, including the anchors'.** The two rewritten anchors kept tolerances never measured (headroom about 1e12 and 1e5); they are measured, recorded and tightened. Oracles take their times from `torch.linspace` and `half_open_grid` rather than the library's sampling schemes.

## Decided after the second review

*Every in-change finding in scope, decided by the user.*

**A slice count is any true integer.** `operator.index(nt)` accepts Python, numpy and 0-d torch integers and refuses floats; a boolean is still refused. `isinstance(nt, int)` had refused `numpy.int64(4)`, contradicting "any integer slice count".

**Emptiness is checked first, in both directions.** An empty basis is refused before the time-dependent branch reads `time_size`, which raised `IndexError` on an empty time-dependent basis. `to_time_independent` gains the forward conversion's refusals: fewer than two slices, which silently produced a field constant in time, no spatial axis, and no coefficients.

**A per-axis sampling must name every axis.** Resolving a per-axis sampling whose length differs from the field's axis count raises a named error, so `axis_sampling` delivers what its name promises. A 3-tuple on a two-axis field had passed through and handed a one-axis field two schemes.

**A no-op conversion shares its coefficients.** ~~It clones them.~~ *Reversed by the user before implementation:* a clone doubles the coefficient grid's memory on every no-op call, no caller mutates a converted field in place, and `copy()` shares storage everywhere else. The scenario is reworded from "an independent copy" to "a new field over the same coefficients", matching `copy()`, and the shared storage is pinned as stated behaviour.

**Accuracy is stated for exact evaluation; the strategy is honoured.** The user chose to keep honouring the caller's strategy, memory budget included, over forcing the exact path. The spec scopes the rounding and interpolation claims to exact evaluation. A new scenario checks that a converted field matches the same sampling and transform under the same strategy. Task 5.8(c) was wrong: under `EvaluationStrategy(memory_budget_mb=1e-6, allow_approximate=True)` at modes (16, 16), dropping the strategy moves the coefficients by 2.08e-3, so that mutant is not equivalent.

**One equivalence is conditional.** Task 5.8(a), dropping the spatial sampling, is equivalent only because a closed spatial axis, the one admissible-by-type scheme that would tell, is itself broken on a Fourier basis (below).

**An intended duplicate.** `test_to_time_independent_values_match_analytic` checks a narrower case of the fuzzed inverse test with the same `numpy.interp` oracle. It is kept as a literal anchor, as the user decided for the analytic anchors.

**Deferred to their own changes.** Two pre-existing defects surfaced and are left out of this change. A `FourierBasis` accepts `ClosedUniform` sampling but is inconsistent with itself under it: `transform(get_values())` differs from its own coefficients by 6.7. And `copy()` drops a configured precision for its other callers. `from_config` passing a dtype mismatch between tensor and config through unchanged goes with the second.

## Risks / Trade-offs

- **Something relied on the half-open placement.** → `to_time_dependent` is called by the library only through the Burgers generator's own copy, which this change leaves alone; the suite runs in full before and after, and any test whose expectation moves is named in the commit.
- ~~**Evaluating at explicit points costs more than a transform.**~~ *Stale after the amendment above:* the conversion goes through the inverse transform with a closed time axis, so there is no per-point evaluation; the suite's wall time is still recorded before and after.
