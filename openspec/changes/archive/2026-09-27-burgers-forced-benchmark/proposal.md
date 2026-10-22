# A forced-Burgers row whose forcing is a prior, not a residue

## Why

The benchmark's `f -> u` row is manufactured-solution only. MMS picks a random
band-limited `u` and *defines* `f` as its Burgers residual, so the forcing the
operator learns from is "whatever a random `u` left behind" — not a distribution
anyone chose, and not one a reader of the table recognises. A reader seeing
`f -> u` reasonably assumes the forcing was drawn and the solution computed; here
it is the other way round.

`time-dependent-forcing-in-numerical-burgers`, since archived as the main spec
`burgers-time-dependent-forcing`, is what makes the honest version buildable:
until it landed, `Burgers.generate` raised `NotImplementedError` for any forcing
that was not a constant.

## What Changes

- One new dataset in `scripts/run_benchmark.py`: `u0 = 0`, forcing drawn as a
  Gaussian random field over `(time, space)`, solution integrated numerically by
  `_solve_spectral` and kept `time_dependent=True`. The operator learns `f -> u`
  where `f` was sampled first.
- Registered as `"project Burgers forced (f->u, GRF forcing, from rest, pilot: t
  periodic)"` under
  key `burgers-forced`, beside the existing `"project Burgers MMS (f->u)"`, so
  the two forcing distributions appear in one table and the comparison is
  explicit rather than implied.
- **Landed as a pilot.** A time-dependent output cannot be scored yet, so the
  row scores `u` as a field periodic in time over `[0, T)`, only at times where
  that is exact, and leaves `u(T)` unscored. A change beneath will make
  time-dependent outputs scorable, and the row will be regenerated then.
- **A reported number is added.** Every change in this stack so far has been able
  to claim that no benchmark row moves. This one cannot: it adds a fifth row. The
  claim it can make instead is that the existing four are unchanged, and it must
  be verified rather than asserted.

## Capabilities

### New Capabilities

None. The requirement this change carries is about what the comparison covers and
what a reader may conclude from it, which `operator-comparison` already owns.

### Modified Capabilities

- `operator-comparison`: its purpose is what the harness guarantees and what the
  reported numbers license a reader to conclude. A second row whose forcing is a
  drawn prior rather than a derived residue changes exactly that, so the
  requirement lands here rather than in a capability of its own.

*Alternative considered:* a new `benchmark-datasets` capability describing the
inventory of rows. Rejected for one dataset — it would sit adjacent to
`operator-comparison` describing the same table, and the instruction against
near-duplicate capability names applies. If `benchmark-reproducibility` later
needs to specify dataset definitions in their own right, that is the moment to
split them out, with two changes' worth of material rather than one.

## Impact

- `scripts/run_benchmark.py`: one `burgers_forced_case` builder, whose sizes
  are keyword arguments defaulting to what `quick` selects, and one
  `DatasetSpec` entry. It must target the current harness API —
  `Split.partition(input_field, output_field, n_train, n_sensors)`,
  `BenchmarkCase(split, query)` and the `CaseBuilder` signature
  `(seed, quick, device)` that `follow-the-stated-device` gave every builder —
  not `OperatorDataset.from_fields`, which an earlier change replaced.
- `tests/`: the four scenarios below need tests naming them, per the
  `test-oracle-convention` requirement that every stated scenario is verifiable.
- `results/bench_results.csv`: gains a row. The four existing rows must be shown
  unchanged against a base run, since library behaviour beneath this change has
  moved and a diff argument no longer settles it. Both runs state the same
  `--device`: a table is reproducible only against the device it was built on.
- No library source changes. `SpectralSVR/` is untouched; the capability this
  depends on already landed.
