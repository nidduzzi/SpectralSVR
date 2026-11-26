## Context

See proposal.md - Why for the five gaps. The current output, verified:

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
- Finding why DeepONet fails on the MMS row. The reference protocol is adopted and measured; whatever r2 results is reported, not chased.

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

### The harness is not run-to-run deterministic, and that is not what this change fixes

Measured while implementing `dtype-seam`: six consecutive runs of identical code in the same tree vary by up to
`2.51e-05` relative. Dataset generation is bit-identical and an isolated fit is bit-identical across processes, so
the variation is neither seeding nor the model; pinning `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 cuts it to
`6.15e-07`, pointing at multi-threaded reductions in the solve.

That floor is small enough not to matter here, and this was checked rather than assumed. Between-seed spread over
seeds 0-4 runs `26,010x` to `180,953x` the noise floor across the four rows, with `mse` moving 242% and 454%. So the
spread this change reports is seed variation, not measurement noise, and the two need not be separated. Stated
because a reader can otherwise reasonably ask whether the reported spread is real.

The figures are from one machine and belong in the output's provenance, not in the library. Re-measure them under
this change's own protocol rather than carrying them forward as constants.

`2.51e-05` is an upper bound observed on some comparisons, not a floor every comparison reaches. Measured later in
this stack: `explicit-band-type`'s base-against-head comparison came back at `2.512e-05`, while two separate
base-against-head comparisons for `operator-harness-spec` were byte-identical at `0.000e+00`. Which rows move, and
whether any move at all, shifts between batches. Treating `2.51e-05` as a constant to compare against would make a
clean comparison look anomalous and a noisy one look acceptable; it is a bound to read a difference *against*, not a
value to expect.

### DeepONet is seeded before its network is built

`dde.config.set_random_seed` moves above the `DeepONetCartesianProd` constructor, so the seed governs the initial weights as well as training. Measured in fresh processes on the quick MMS split at 2000 iterations: the current order gave r2 -0.840 and -1.716 on two cold runs and -0.119 after an earlier fit; the new order gave r2 -0.118985 with identical prediction bytes in all three positions, on cuda, without `torch.use_deterministic_algorithms`. Enabling deterministic algorithms as well gave the same bytes, so it is not adopted at this size. The position-independence scenario is what guards it, since a longer run or a larger split may reach a nondeterministic kernel that this one did not.

*Alternative rejected:* seed once per process. It leaves every fit's weights depending on what ran before it, which is the defect.

### DeepONet is trained by a named recipe taken from reference code

The recipes come from the code the reference results were produced with, not from a paper's prose. Two apply:

- **Lu 2021 / deepxde example**, for the antiderivative rows. Branch `[m, 40, 40]`, trunk `[d, 40, 40]`, ReLU, Glorot normal, Adam at 1e-3, no normalisation. Sources: deepxde `examples/operator/antiderivative_aligned.py` (10,000 iterations) and `lululxvi/deeponet` `src/deeponet_pde.py`, problem `ode` (50,000 epochs). This is the current configuration.
- **Lu 2022 Burgers**, for the three Burgers rows. Branch `[m] + [128]*4`, trunk `[d] + [128]*3`, tanh, Glorot normal, trunk features `cos` and `sin` of 2πx and 4πx for each periodic spatial coordinate (a time coordinate is passed unchanged), outputs normalised point by point as `outputs * std + mean` from the training targets, no input normalisation, Adam at 1e-3 with inverse-time decay `(iterations // 5, 0.5)`. Source: `lu-group/deeponet-fno` `src/burgers/deeponet.py`, which trains for 500,000 epochs.

The scaling takes the reference's form, `outputs * std + mean` applied to the network's output, with the network trained against unscaled targets. It never divides, so a point where the training targets are constant, with a standard deviation of zero, is pinned to their mean. That is exact for the forced row's twelve `t = 0` points, which start from rest. *Corrected from this design as first written,* which assumed the targets were divided by their deviation and planned a scale of 1 at such points; the reference code does not divide. The statistics come from the training split only, and because they belong to the training points, a model fitted by this recipe refuses a query at any other points.

Adoption is decided per Burgers row on five seeds, as the reference reports five trials. The current configuration and the Burgers recipe are each measured at 10,000 and 50,000 iterations. The Burgers recipe is adopted on a row where its better mean r2 is not below the current configuration's better mean by more than the larger of their two standard deviations; the longer run is kept only where it beats the shorter by the same margin, since it multiplies regeneration cost. Otherwise the current configuration stays, and the table records the Burgers recipe's measured result as the departure. 500,000 epochs would take about ten minutes per fit and some twenty-five hours per regeneration, so the length is a stated departure in every case.

This is not tuning. Nothing is searched; every choice is a reference's, cited in the table, and the comparison between two fixed recipes is recorded whichever way it falls.

*Why the first plan was dropped.* It took He initialisation (§3.1.5), input normalisation and point-by-point output normalisation (§3.4) from the paper's text. Measured cumulatively at seed 0, full size and 10,000 iterations, with the r2 move under 1e-7 and 1e-6 sensor shifts in brackets:

| dataset | seeded baseline | + He init | + input norm | + per-point output norm | + second-moment rescale |
| --- | --- | --- | --- | --- | --- |
| antiderivative | 0.616 (0.04) | 0.012 | 0.397 | 0.477 (0.02) | 0.585 |
| burgers-mms | -6.12 (4.87) | -253 | -1186 | -11.79 (2.64) | -7.50 |
| burgers-numerical | -1.67 (0.08) | -7.69 | -22.8 | -0.19 (0.01) | -0.94 |
| antiderivative-grf | 0.9991 (0.00) | 0.978 | 0.995 | 0.998 (0.00) | 0.9989 |
| burgers-forced | 0.801 (0.03) | -22.4 | -10006 | -1.52 (0.67) | -0.29 |

He initialisation alone wrecked every row, and input normalisation made it worse. Neither appears in any of the reference code: its ReLU networks use Glorot initialisation and it never normalises inputs. The second-moment rescaling by O(1/√p) of §3.1.5 is not in the reference Burgers code either, and the paper itself reports that vanilla DeepONet "works well even without scaling", so it is not pursued. The grid also shows that one seed cannot rank recipes: the seeded baseline's MMS r2 moves 4.87 under a 1e-6 shift. Hence five seeds.

*Alternative rejected:* keep the current configuration everywhere and state its departures. It is the reference recipe for the antiderivative rows, but for Burgers a reviewer can point to the fair comparison's code, and that objection is answered by measurement, not by a caveat.

### A baseline predicts on the query's device

`FittedDeepONet.predict` returns its result on the device of the points it was queried at, matching `sensors-follow-the-stated-device`'s rule that derived work follows placed data. Today it lands on the process default, which deepxde switches to cuda at import.

### Run-to-run spread is measured per operator

The SpectralSVR noise floor was measured under `dtype-seam`; DeepONet's never was. Each operator is run repeatedly at one fixed seed, first in a process and after other fits, and the spread is recorded per dataset beside the seed spread. The table's documentation states both, so a DeepONet seed spread is not read as method variation where repeated runs vary as much.

## Risks / Trade-offs

- **Every DeepONet row moves.** Seeding changes every fit, and the Burgers recipe changes the rows it is adopted on. → Each move is measured and recorded against the element that caused it, and the table cites the protocol, so the discontinuity is documented rather than discovered.

- **DeepONet may still fail on MMS.** A reference recipe is not a promise of accuracy on a manufactured operator.

- **The Burgers recipe costs more to measure.** Wider networks at 50,000 iterations over five seeds and five datasets run for about three hours. → The measurement runs once, in the background, and its raw rows are kept with the change. → The row is reported as it comes out, with its fixed-seed spread beside its seed spread.

- **Regeneration cost multiplies by the seed count.** DeepONet trains for ten thousand iterations per dataset per seed. → The long format means a partial run can be appended rather than restarting, and the quick mode stays available for smoke checks.

- **The table gains columns and changes shape.** Anything reading the current CSV by position breaks. → The notebook is the only consumer and is updated in this change; the file is named as a record whose schema is part of the contract.

- **Two relative-error columns invite the wrong one being quoted.** → Both are named explicitly and the documentation states which convention matches the neural-operator literature.

- **Provenance columns are only as good as what is recorded.** A dirty working tree makes a revision meaningless. → The recorded revision notes whether the tree was clean, so a number produced from uncommitted code is visible as such.

## Migration Plan

1. Add the per-sample relative L2 to the metric suite beside the existing entries; confirm no existing metric value changes.
2. Seed DeepONet before its network is built, return its predictions on the query's device, add the Lu 2022 Burgers recipe beside the current one, and adopt it per row by the five-seed criterion.
3. Add provenance capture and the long output format to the benchmark script; confirm a single-seed run reproduces today's SpectralSVR numbers on the columns that already existed to within the harness's measured run-to-run noise of `2.51e-05` -- exact reproduction is not achievable and asserting it would fail for reasons unrelated to this change. DeepONet rows are expected to move, from step 2, and are recorded rather than compared against that floor.
4. Add seed aggregation and the `just` recipe.
5. Point the notebook at the committed table.
6. Regenerate over the full seed set, after the three dependencies have landed, and commit the result with its provenance.

Rollback is a revert plus restoring the previous table from history.

## Open Questions

- Whether a paired comparison across matched seeds should be reported rather than independent spreads. Deferrable: the long format preserves the data either way, so this can be decided when the numbers are in hand.
