## Context

See proposal.md — Why. The design-relevant state:

```
burgers_mms_case                scripts/run_benchmark.py
  (seed, quick, device) -> BenchmarkCase            <- the shape to copy
  Split.partition(f, u, n_train, n_sensors=modes)
BenchmarkCase, DatasetSpec      scripts/run_benchmark.py
  (split, query); (label, build)
grf_coeff                       scripts/run_benchmark.py
  periodic squared-exponential GRF sampler, already used in 1D
Burgers._forcing_at             SpectralSVR/problems/burgers.py
  accepts a Basis field spectral in time; rejects a per-slice field
```

| fact | consequence |
| --- | --- |
| `_forcing_at` accepts a field with `time_dependent=False` and rejects `True` | the sampled forcing must be built spectral in time, which is what `FourierBasis.generate` over a 2D domain produces |
| `Split.partition` slices both fields and builds two `OperatorDataset`s | a 2D-domain input field must survive `from_fields`' sensor sampling |
| `SpectralSVROperator.fit` passes `u_time_dependent=dataset.u.time_dependent` | a time-dependent output field is already a supported shape |
| the four existing rows came from a base run at this branch's parent | "the others are unchanged" is measurable, not arguable |

## Goals / Non-Goals

**Goals:**

- A `f -> u` row whose forcing was drawn before the solution existed.
- Leave the four existing rows untouched, and show it rather than argue it.

**Non-Goals:**

- Changing any library source. The capability this needs already landed.
- Tuning either operator for the new row. The comparison holds tuning fixed and
  says so; this row inherits that, and inherits the caveat with it.
- A two-input operator taking both a random initial condition and a random
  forcing. Recorded as an open question below.

## Decisions

### The forced dataset starts from rest and samples a time-space GRF

`u0 = 0`, forcing `f ~ GRF(time, space)` built by
`FourierBasis.generate(n, (mt, ms), random_func=grf_coeff(...), domain=((0, T), (0, L)))`
— the same sampler the antiderivative-GRF case uses, in two dimensions. Length
scales are dataset parameters recorded in the `DatasetSpec` label. The output is
the time-dependent `u` from `_solve_spectral`, kept as `time_dependent=True`.
Like every builder since `follow-the-stated-device`, it takes the device it is
placed on: its generator is built there and `device` is passed to each
`generate`, so no stage follows the process default.

Starting from rest is what makes the operator `f -> u` alone. With a random
initial condition as well, the map takes two functions, and DeepONet's branch net
takes one — the comparison would stop being like-for-like on the very row added
to make it honest.

*Alternative rejected:* converting `u` to a fully spectral field with
`to_time_independent` for cheaper evaluation. A forced solution from rest is not
periodic in time; a Fourier transform along `t` would ring at the boundaries.

### The dataset is built through `Split.partition`, not two loose datasets

The design this change inherited said
`OperatorDataset.from_fields(f, u, n_sensors)`. That shape no longer exists at
the harness boundary: `operator-harness-spec` made overlapping train/test splits
unconstructible by giving `benchmark` a `Split` that only `Split.partition`
builds. Every existing case already reads that way.

Recording it because the inherited note was wrong, not because the choice is
interesting — a task written against the old API would have compiled a dataset
the benchmark cannot take.

### The forced builder states its sizes

`burgers_forced_case` takes keyword `n_train`, `n_test` and `modes`, each
defaulting to what `quick` selects. It still satisfies `CaseBuilder`, so the
registry and the CLI are unaffected, and tests can fuzz sizes down to the
smallest the harness accepts instead of sampling two fixed configurations. Only
the new builder gains this. Widening `CaseBuilder` for every case is a separate
decision that nothing here needs.

### "Adding a row" is modelled as a sequence

The claim that adding a dataset leaves the others alone constrains what stays
true across builds. It is tested with a state machine that builds registered
cases in any order and repoints the process default between builds. A single
before-and-after test would reach only one of those orders. The base run in the
tasks is the scenario's other half, over reported rows, and stays a
verification.

### The label carries the forcing's provenance

`"project Burgers forced (f->u, GRF forcing, from rest)"`, beside
`"project Burgers MMS (f->u)"`. Both appear in the table, and the difference a
reader needs — drawn prior versus derived residual — is in the label rather than
in a footnote they may not reach.

## Risks / Trade-offs

- **Time-dependent output fields must evaluate correctly at 2D query points in
  `OperatorDataset.targets`.** The existing `u0 -> u(T)` case sidestepped this by
  slicing to spatial fields, so this row is the first to exercise it. → A task
  measures `targets(query)` against direct evaluation of the stored coefficients
  at grid points, records the gap, and sets the bound from the measurement. If
  time interpolation error is visible, the dataset's query grid is restricted to
  output-grid times and the reason recorded in the `DatasetSpec` docstring.

- **The GRF length scale in time controls how much the forcing varies within a
  step.** Too smooth and the stage-time behaviour is under-exercised; too rough
  and the solution is dominated by forcing noise. → The length scale is a
  recorded dataset parameter. The correctness argument does not rest on it: the
  round-trip anchor beneath this change uses MMS forcing, not the GRF.

- **A new row cannot claim what the others claimed.** Four changes in this stack
  proved "no number moves" from the diff. → This one proves the weaker, true
  thing instead: the four existing rows are byte-compared against a base run on
  the same `--device`, and the fifth is reported as new. In-process, a test also
  builds every existing case before and after the new one and requires them
  bit-identical, which catches a builder leaking state into its neighbours. That
  is how `deepxde` once reset the default device mid-run.

- **The row is generated before `benchmark-reproducibility` defines provenance.**
  → That change sits directly above and will regenerate the table anyway; this
  row is built to be regenerated, not to be final. Its seed, parameters and
  `--device` are recorded so the regeneration is reproducible rather than
  approximate. The device matters: a generator draws different numbers on a
  different device from the same seed, so `--device cuda` builds a different
  dataset. *Corrected:* this risk used to name `align-device-resolution` as
  settling where fitting runs. That change was withdrawn, and
  `follow-the-stated-device` settled it by adding `--device`.

## Migration Plan

One builder, one registry entry, and the tests in tasks section 3. Nothing persists and no library
source changes, so rollback is a revert. The `results/` CSV is generated data and
gitignored; the row appears on the next regeneration.

## Open Questions

- Whether the forced dataset should also be offered with a random initial
  condition, as a two-input operator, once a baseline that accepts two input
  functions exists. Deferrable: it changes neither this change's requirements nor
  its tasks, and no such baseline exists today.
