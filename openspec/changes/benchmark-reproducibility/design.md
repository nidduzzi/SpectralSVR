## Context

See proposal.md - Why for the four gaps. The current output, verified:

```
dataset,operator,fit_seconds,mse,rmse,mae,r2,smape,rse,rrse,pred_nan_sum
```

One row per dataset and operator, seed 0, no provenance. `DEFAULT_METRICS` in `utils/__init__.py` defines `rrse` as the square root of a *global* `SSE/SST` — one ratio over all elements — which is the statistic the ETDRK4-era fix deliberately chose over torchmetrics' per-column mean. That decision stands; the gap is that the field's convention is a third thing again, per-sample and then averaged.

Constraints inherited from elsewhere: `align-spectral-residual-with-solver` moves the Burgers MMS row, `operator-harness-spec` states what the table already guarantees, and the forcing change adds a dataset that belongs in the same run.

## Goals / Non-Goals

**Goals:**

- A table whose every number carries a spread and a provenance.
- An error statistic a reader of the neural-operator literature will read correctly.
- One command to regenerate, one file of record.

**Non-Goals:**

- Tuning. Untuned-on-both-sides is a stated limitation of the harness; removing it needs a protocol, a budget and its own change. Adding seeds without tuning is still an improvement, and conflating the two would make this change unlandable.
- New baselines or datasets. FNO belongs with its own adapter change.
- Changing `rse` or `rrse`. Their definitions were settled deliberately; this change adds beside them.

## Decisions

### Long format on disk, aggregation at read time

The file records one row per dataset, operator and seed. The centre and spread are computed when the table is presented rather than baked into the file.

Keeping the raw runs means a later question — a different centre statistic, a per-seed outlier, a paired test between operators on matched seeds — can be answered without re-running anything, which at ten thousand DeepONet iterations per seed is the expensive part. An aggregated file forecloses that.

*Alternative rejected:* store only the aggregate. Smaller and directly presentable, but discards exactly the data that makes the aggregate defensible.

### Provenance is per run, not per file

Device, precision, library versions and revision are columns, not a header comment. A results file may accumulate runs from more than one session, and a header can only describe one of them. Columns also survive concatenation and filtering, which a header does not.

### The new statistic is named for what it is

The per-sample relative L2 is reported under a name that says so, beside `rrse` rather than replacing it. Both stay in the metric suite so that any table generated before or after this change remains comparable on the columns it already had.

*Alternative rejected:* redefine `rrse` to the per-sample convention. It would silently change the meaning of a column that already appears in a committed table, which is the failure mode this change exists to prevent.

### Timing is reported per operator with what it covers

Rather than deleting the column or trying to normalise it, each operator declares what its fit does — a direct solve, or a fixed iteration count — and that is recorded alongside the time. The comparison then reads as two facts rather than one ratio.

*Alternative rejected:* normalise by iteration count. Invents a comparison that does not exist; a solve has no iteration count.

### Seed count is fixed and small

Enough seeds for a spread to mean something, few enough that a full regeneration stays feasible given DeepONet's training cost per run. The set is recorded in the file rather than left to a default that might change, so an old table stays interpretable.

## Risks / Trade-offs

- **Regeneration cost multiplies by the seed count.** DeepONet trains for ten thousand iterations per dataset per seed. → The long format means a partial run can be appended rather than restarting, and the quick mode stays available for smoke checks.

- **The table gains columns and changes shape.** Anything reading the current CSV by position breaks. → The notebook is the only consumer and is updated in this change; the file is named as a record whose schema is part of the contract.

- **Two relative-error columns invite the wrong one being quoted.** → Both are named explicitly and the documentation states which convention matches the neural-operator literature.

- **Provenance columns are only as good as what is recorded.** A dirty working tree makes a revision meaningless. → The recorded revision notes whether the tree was clean, so a number produced from uncommitted code is visible as such.

## Migration Plan

1. Add the per-sample relative L2 to the metric suite beside the existing entries; confirm no existing metric value changes.
2. Add provenance capture and the long output format to the benchmark script; confirm a single-seed run reproduces today's numbers on the columns that already existed.
3. Add seed aggregation and the `just` recipe.
4. Point the notebook at the committed table.
5. Regenerate over the full seed set, after the three dependencies have landed, and commit the result with its provenance.

Rollback is a revert plus restoring the previous table from history.

## Open Questions

- Whether a paired comparison across matched seeds should be reported rather than independent spreads. Deferrable: the long format preserves the data either way, so this can be decided when the numbers are in hand.
