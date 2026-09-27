## Context

See proposal.md - Why. Four facts, measured before proposing, shape the
approach:

| fact | consequence |
| --- | --- |
| `FourierBasis.evaluate(time_dependent=True)` returns `out[:, i, j] = u(t_i, x_j)`, a grid over `Q` times by `Q` points | paired query rows need paired evaluation; the grid's diagonal costs `O(Q²)` |
| that evaluation is exact at every slice time, including `t = t1`, and linear between slices (index `(t - t0)/(t1 - t0) * (T - 1)`) | the field's own convention for reading between slices already exists, and paired evaluation must use it, not invent another |
| `SpectralSVR.train` stores spatial `modes` only, and flattens `(n, T, *modes)` coefficients | `forward` cannot restore the time axis, because the slice count is not recorded |
| `SpectralSVR.forward` evaluates on `self.config.domain`, which carries no time interval of the training field | the time interval must reach `forward` explicitly |

## Goals / Non-Goals

**Goals:**

- One convention for reading a time-varying field at a query row, shared by the
  targets and by SpectralSVR's prediction, so the two cannot disagree.
- Cost linear in the number of query rows.
- Time-invariant evaluation untouched, and shown bit-identical.

**Non-Goals:**

- Changing how a time-varying field is interpolated between slices. The
  existing linear convention stands.
- Changing the grid form of `FourierBasis.evaluate`. Existing callers keep it.
- Sensor placement, which is the next change.
- DeepONet, which already predicts at arbitrary `(t, x)` rows.

## Decisions

### Paired evaluation lives with the basis, beside the grid form

A time-varying coefficient tensor `(n, T, *modes)` is evaluated at paired rows
`(t_q, x_q)` in three steps:

1. Evaluate the spatial basis at the `x_q`.
2. Sum each slice against it, giving `(n, T, Q)`.
3. Interpolate each column `q` linearly between the two slices around `t_q`,
   with the same index map as the grid form.

That gives `(n, Q)` at a cost of `O(n · T · Q)`. It sits in the basis because
the targets and the model both hold coefficients in the basis's layout and must
read them the same way.

*Alternatives rejected:*

- Taking the diagonal of the grid. It computes `Q²` values to keep `Q`.
- Grouping rows by distinct time and evaluating a grid per group. It is only
  cheap when queries happen to be tensor products, and it adds a second code
  path whose agreement with the first would itself need proving.

### A query row states its time first

For a time-varying output, a query row is `(t, x_1, ..., x_d)`. That is the
layout the forced row already uses, and it matches the field's `domain`, whose
first interval is time. A row carrying only the `d` spatial coordinates is
refused, with a message naming the layout. Guessing a time would return values
for a time the caller did not state.

### The model records what it needs to read its output back

At `train`, `SpectralSVR` records the output's slice count alongside its spatial
`modes`. `forward` reshapes the predicted coefficients to `(n, T, *modes)` and
evaluates them at the paired rows. The time interval reaches `forward` through
its existing `domain` argument. `FittedSpectralSVR` captures the training
field's domain at `fit` and passes it on every `predict`. The model is not made
to infer a time interval from its configuration, which never held one.

### The forced row drops the pilot

`burgers_forced_case` keeps `u` time-dependent and queries 12 times across
`[0, T]` inclusive, by 12 points in space. `periodic_in_time` and its test are
deleted, and the label loses `pilot: t periodic`. The row is regenerated and
reported as moved deliberately, with the pilot's numbers kept for comparison.

## Risks / Trade-offs

- **Linear interpolation between slices is part of what is scored.** At an
  off-slice time, the target is the interpolant, not the solver's value at that
  time. → Query times are placed on slice times, as the pilot's were, so every
  target is a stored value. A test checks that placement, and the
  interpolation's own accuracy is out of this change's scope.
- **A second evaluation path could drift from the grid form.** → An invariant
  test requires paired evaluation at every `(t_i, x_j)` to equal the grid form's
  `out[:, i, j]` exactly. Enumerated mutation over the new path checks that the
  test can fail.
- **Recording the slice count adds state to a fitted model.** → It is set only
  at `train`, in the same place as `modes`, and `forward` refuses when it is
  absent. Nothing reads it before a fit.
- **The forced row's numbers move.** → That is the point. The other rows are
  compared bit-for-bit against a base worktree with its own environment on the
  same `--device`. The first DeepONet fit of a run is excluded from
  bit-identity, as `burgers-forced-benchmark` measured it varying between runs
  of identical code, and is compared against its measured spread instead.

## Migration Plan

1. Paired evaluation in the basis, with its invariant and analytic tests.
2. `targets` and `forward` use it; time-invariant paths shown bit-identical.
3. The forced row drops the pilot and is regenerated.

Rollback is a revert of all three, in reverse order.
