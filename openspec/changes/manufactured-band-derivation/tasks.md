## Execution order

```
1 derive the cutoff ── 2 expose and validate ── 3 anchors ── 4 verify
```

A chain; nothing depends on a later section. Section 1 leaves the derived band as
the only path, so the suite is green before the argument exists.

## 1. The derived cutoff

- [x] 1.1 Replace `band_limited = tuple(max(1, m // 3) for m in modes)` (`burgers.py:337`) with the largest cutoff satisfying `2K <= m / 3`, stated through the cutoff form of the band type; verify the resulting kept set is `-K … +K` and therefore odd-sized at every mode count
- [x] 1.2 Verify the band got **wider**, not narrower, and record the old and new width at mode counts 8, 12, 16 and 32. A count-based band `B` has highest frequency `B//2` and spends a slot on asymmetry; a cutoff with the same highest frequency keeps `2K + 1` modes. At 8 modes the old band is 2 wide with highest frequency 1 and the new one should keep 3 with the same highest frequency. A narrower result means the derivation is wrong, not conservative
- [x] 1.3 Verify the manufactured field is conjugate-symmetric **with `resize_modes`' boundary rule disabled** — temporarily, in the measurement, not in the code. It reads `7.859e-01`, `2.957e+00`, `0.000e+00`, `7.392e+00` at modes 8, 12, 16, 32 before this change; it must read `0.000e+00` at all four after. Asserting it with the rule enabled proves nothing, since `utils-encoding-cleanup` already guarantees that
- [x] 1.4 Verify the continuous residual stays at rounding level at every mode count, so the wider band did not breach the threshold; compare against `MANUFACTURED_CONTINUOUS_RESIDUAL` and record the new values
- [x] 1.5 Run `just check`; verify clean

### Measured, section 1

`Burgers.widest_manufactured_band(m)` returns `WavenumberCutoff(m // 6)`, the
largest `K` with `2K <= m / 3`.

| mode count `m` | 8 | 12 | 16 | 32 |
| --- | --- | --- | --- | --- |
| old band, count `max(1, m // 3)` | 2 | 4 | 5 | 10 |
| old highest frequency `B // 2` | 1 | 2 | 2 | 5 |
| new cutoff `K` | 1 | 2 | 2 | 5 |
| new band, count `2K + 1` | 3 | 5 | 5 | 11 |
| `2K` vs threshold `m / 3` | 2 ≤ 2.67 | 4 ≤ 4.00 | 4 ≤ 5.33 | 10 ≤ 10.67 |

The band is wider at 8, 12 and 32, and equal at 16 — never narrower, which is
what 1.2 required. The highest frequency is unchanged at every count, so the
extra mode is the slot the count-based band was spending on asymmetry, exactly
as the design predicted. `m = 16` is equal because `16 // 3 = 5` was already
odd; it is the same case the proposal isolates as its control.

Conjugate symmetry, Hermitian error with `resize_modes`' even-source boundary
repair removed from the padding path (removed in the measurement, not the code),
swept over mode counts 6–39 plus 48 and 64, seeds 0–4, 180 cases:

| | worst Hermitian error | cases already at zero |
| --- | --- | --- |
| old band | 1.366e+01 | 85 / 180 |
| new band | **0.000e+00** | 180 / 180 |

The new band reads exactly zero in every case, so the field is real as generated
rather than as repaired. The old band reads zero in the 85 cases where `m // 3`
happens to be odd — parity is the whole of it, as the proposal states.

**Discrepancy, recorded rather than smoothed over.** Task 1.3 quotes the
pre-change readings as `7.859e-01`, `2.957e+00`, `0.000e+00`, `7.392e+00` at
modes 8, 12, 16, 32. Those exact magnitudes did not reproduce: they depend on
the generator seed and the sample count, and a scan of seeds 0–7 matched none of
them on all four counts. What *is* structural, and does reproduce at every seed,
is the pattern the numbers were recorded to show — nonzero at 8, 12 and 32,
exactly `0.000e+00` at 16, and exactly `0.000e+00` at all four after the change.
The after-state, which is what this task asserts, is exact and seed-independent.

Continuous residual, `nu = 0.05`, seed 3, `n = 1`, evaluated off the sample grid
at `nq = 96` against `MANUFACTURED_RESIDUAL_TOLERANCE = 5e-6`:

| mode count | recorded before | measured after |
| --- | --- | --- |
| 8 | 2.151e-07 | 2.565e-07 |
| 12 | not recorded | 2.175e-07 |
| 16 | 2.099e-07 | 1.999e-07 |
| 32 | 2.623e-07 | 2.954e-07 |

All at rounding level, more than an order of magnitude inside the tolerance, so
the wider band did not breach the de-aliasing threshold. `just check`: 303
passed, ruff/mypy/ty clean.

## 2. The band as a stated argument

- [x] 2.1 Add an optional band parameter to `mms_solution`, defaulting to the derived cutoff; verify every existing call site is unaffected
- [x] 2.2 Thread it through `Burgers.generate` as an explicit named parameter beside `nu` and `solver` — **not** through `**kwargs`, which `Burgers.generate` never reads (`ty` reports it unused at `burgers.py:50`) and would silently discard it. Verify the parameter is explicit rather than absorbed: a band stated through `Burgers.generate` reaches `mms_solution` and changes the generated field's width against the default, while the same value passed under an unrecognised keyword leaves the width at the default — the second case is the silent discard this task exists to prevent
- [x] 2.3 Validate a stated band against `2K <= m / 3` and raise when it overruns, naming both the stated bandwidth and the widest the mode count admits; verify the message contains both numbers
- [x] 2.4 Verify the refusal is justified by measurement rather than caution: a band at `m//2` leaves a continuous residual of `1.50e-1` and `m//2 - 1` leaves `7.56e-2`, against `3.64e-7` inside the threshold. Record that an overrunning pair is still self-consistent, which is why refusing at generation is the only place it can be caught
- [x] 2.5 Make the numerical branch of `Burgers.generate` reject a band, mirroring how the manufactured branch already rejects `solver` at `burgers.py:84`; verify the error says why it cannot act rather than ignoring it
- [x] 2.6 Verify a band that is structurally impossible — not positive, or wider than the grid — is refused separately from one that merely overruns the threshold, since they are different errors
- [x] 2.7 Run `just check`; verify clean

### Measured, section 2

The parameter is typed `RealBand | None`, not `Band | None`. `ExactModeCount`
cannot describe a real field and the spec requires a band that cannot select one
member of a conjugate pair, so the type refuses it at the call site rather than a
runtime check reporting it later.

`band=None` and `band=widest_manufactured_band(m)` produce `torch.equal`
coefficients, so every existing call site is unaffected (2.1).

**The parameter is explicit, not absorbed** (2.2). At `m = 16`, default width 5:

| call | width |
| --- | --- |
| `generate(..., band=WavenumberCutoff(1))` | 3 |
| `generate(..., bandwidth=WavenumberCutoff(1))` — unrecognised keyword | 5 |

The named parameter reaches `mms_solution` and narrows the field; the same value
under an unrecognised keyword is swallowed by `**kwargs` and leaves the default.
The second row is the silent discard the task exists to prevent.

**The two refusals are different errors** (2.3, 2.6). Structural is checked
first, since a band past the grid also overruns the threshold and the more
specific message is the useful one:

```
WavenumberCutoff(-1)   cutoff must be non-negative, got -1          (the type)
WholeGrid() at m=16    band WholeGrid() keeps |k| <= 8, which needs
                       17 modes on a grid of 16                     (structural)
WavenumberCutoff(20)   ... keeps |k| <= 20, which needs 41 modes
                       on a grid of 16                              (structural)
WavenumberCutoff(3)    band WavenumberCutoff(cutoff=3) keeps |k| <= 3 in 7
                       modes, whose product reaches 6 and overruns the 5.33333
                       the de-aliasing mask keeps; the widest 16 modes admit
                       is |k| <= 2 in 5 modes                       (threshold)
```

The threshold message names both numbers the task required: the stated bandwidth
and the widest the mode count admits.

The numerical branch refuses a band (2.5): *"the numerical branch integrates a
given initial condition and forcing rather than manufacturing a field, so a band
would have nothing to limit"*, mirroring the manufactured branch's refusal of
`solver`. Both rejections now sit in one `_reject_inapplicable` helper — ruff's
C901 put `generate` at 11 > 10 once the second was added inline, and the two are
one rule stated twice, so they were extracted rather than the limit raised.

**The refusal is justified by measurement, not caution** (2.4). `m = 16`,
`nu = 0.05`, seed 3. "discrete" is `spectral_residual`, the de-aliased equation
the pair is built from; "continuous" evaluates every term off the sample grid:

| kept modes | cutoff | `2K` | vs `m/3` | discrete | continuous | |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 2 | 4 | 5.33 | 0.000e+00 | 1.999e-07 | admitted |
| 7 | 3 | 6 | 5.33 | 0.000e+00 | 1.608e-02 | refused |
| 8 | 3 | 6 | 5.33 | 0.000e+00 | 3.631e-02 | refused |
| 9 | 4 | 8 | 5.33 | 0.000e+00 | 7.064e-02 | refused |

The discrete residual reads **exactly zero for every band, including the refused
ones**, because forcing and solver de-alias alike. The continuous residual jumps
five orders of magnitude the moment the band overruns. A pair that checks itself
can never catch this, which is why generation is the only place it can be
refused — the point 2.4 asked to be recorded.

The task quotes `1.50e-1` at `m//2` and `7.56e-2` at `m//2 - 1` against `3.64e-7`.
Those came from a differently parameterised measurement during
`align-spectral-residual-with-solver` — count-based bands and its own seed — so
the magnitudes differ from the table above. The ratio the numbers were recorded
to show is reproduced and larger: five orders of magnitude, not four.

`just check`: 303 passed, ruff/mypy/ty clean.

## 3. Anchors

- [x] 3.1 Add an anchor asserting the generated field is conjugate-symmetric at every supported mode count with the resize boundary rule disabled; verify it fails against the pre-change derivation, which is what makes it a test of this change rather than of the one beneath it
- [x] 3.2 Add an anchor asserting an overrunning stated band raises, and that the widest admissible band does not; verify the boundary between them is exactly `2K <= m / 3` rather than one off it
- [x] 3.3 Update `MANUFACTURED_CONTINUOUS_RESIDUAL` in `tests/problems/test_problems.py` with the re-measured values; verify they are recorded from measurement and that no tolerance was widened to accommodate them
- [x] 3.4 Verify the analytic positive control and the corrupted-forcing negative control both still behave — the control pair is independent of the band, so a change in either indicates this change reached further than intended
- [x] 3.5 Verify `tests/test_oracle_coverage.py` passes with the markers chosen

### Measured, section 3

Three anchors in `tests/problems/test_problems.py`, suite 303 → 306:

| test | oracle | fuzzed |
| --- | --- | --- |
| `test_manufactured_field_is_conjugate_symmetric_as_generated` | `oracle_invariant` | mode counts 6–40, seeds |
| `test_stated_band_narrows_the_manufactured_field` | `oracle_analytic` | `no_fuzz`, every cutoff below the default at `m = 32` |
| `test_a_band_past_the_threshold_is_refused_though_the_pair_is_self_consistent` | `oracle_analytic` | mode counts 12–40 |

**The symmetry anchor discriminates** (3.1). Run against the pre-change
count-based band over mode counts 6–40, it fails on 18 of 35 — all 18 on the
odd-kept-set assertion alone, before the Hermitian measurement is reached. The
other 17 are the counts where `m // 3` is already odd, which is the same
population as the 85/180 in section 1. It also carries its own control: an
even-sized band read through the same repair-disabled path must read nonzero, so
a blind measurement fails rather than passes.

**The refusal boundary is exact** (3.2). Over mode counts 12–40, all 29 have
`2K <= m/3` for the widest admissible cutoff and `2(K+1) > m/3` for the next one
up, so the anchor sits on the boundary rather than one off it. The anchor also
rebuilds the refused pair by hand and asserts its de-aliased residual is below
`1e-10` while its continuous residual exceeds the tolerance — the self-consistency
point, asserted rather than only recorded.

**`MANUFACTURED_CONTINUOUS_RESIDUAL` re-recorded** (3.3) from the section 1
measurement: `{8: 2.565e-07, 16: 1.999e-07, 32: 2.954e-07}`, from
`{8: 2.151e-07, 16: 2.099e-07, 32: 2.623e-07}`. `MANUFACTURED_RESIDUAL_TOLERANCE`
stays `5e-6` and `MANUFACTURED_RESIDUAL_CORRUPTED` stays `1.0e-2`; no tolerance
moved, only the documentary values the assertion messages quote.

**Both controls are unmoved** (3.4), confirming the change did not reach past the
manufactured band:

| control | recorded | measured |
| --- | --- | --- |
| analytic positive, modes 12 | 4.164e-08 | 4.164e-08 |
| analytic positive, modes 16 | 3.887e-08 | 3.887e-08 |
| analytic positive, modes 32 | 3.887e-08 | 3.887e-08 |
| corrupted forcing, negative | detected above 1e-02 | 5.000e-02, detected |

`tests/test_oracle_coverage.py` passes with those markers (3.5).

**Added scope, surfaced rather than absorbed.** Writing these anchors let five
entries leave `tests/test_scenario_coverage.py`'s `EXEMPT` list. Four are this
change's own `IN_FLIGHT` debt, which the tasks imply. The fifth,
`Bandwidth that overruns the grid is rejected or resolved`, was marked `PREDATES`
and is not named in any task — but task 3.2's anchor asserts exactly it, so
leaving the entry would have been a scenario that quietly gained a test, which
the guard's stale-exemption check exists to refuse. Removed, and recorded here
because no task asked for it.

`just check`: 306 passed, ruff/mypy/ty clean.

## 4. Verification

- [x] 4.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [x] 4.2 Verify `resize_modes` is untouched: this change removes the need for its boundary rule in one caller, not in general, and the anchors added by `utils-encoding-cleanup` must still pass unchanged
- [x] 4.3 Verify `Antiderivative` is untouched and gained no band argument — it is linear, forms no product, and needs no de-aliasing headroom
- [x] 4.4 Record that the Burgers MMS row moves and is regenerated by `benchmark-reproducibility`, which sits above this change; do not regenerate the table here
- [x] 4.5 Verify no source file gained an explanatory prose comment; the derivation must read from the band form's name and the validation's error text

### Measured, section 4

`just check`: **306 passed**, ruff, mypy and ty clean (4.1).

Everything this change touched, against the branch beneath it:

```
SpectralSVR/problems/burgers.py    the band derivation, the parameter, the refusals
tests/problems/test_problems.py    three anchors, re-recorded residual constants
tests/test_scenario_coverage.py    one PREDATES entry removed
```

`SpectralSVR/utils/__init__.py` is untouched, so `resize_modes` keeps its
boundary rule and `utils-encoding-cleanup`'s anchors pass unchanged (4.2). This
change removes the *need* for that rule in one caller, not the rule.
`SpectralSVR/problems/antiderivative.py` is untouched and gained no band (4.3) —
it is linear, forms no product, and needs no de-aliasing headroom.

No source file gained an explanatory prose comment (4.5). The derivation reads
from `widest_manufactured_band`'s name and `WavenumberCutoff`'s, and the refusals
read from their error text. One explanatory comment written into the refusal
anchor while drafting was removed and replaced by assertion messages carrying the
same information at the point of failure.

**The Burgers MMS benchmark row moves and is not regenerated here** (4.4).
`u.coeff` is the regression target and reaches the model through `to_real_coeff`,
and the band's width changed at every mode count except those where `m // 3` was
already odd, so the manufactured dataset is a different learning problem after
this change. `f -> u` scores are not comparable across it — the same
discontinuity `align-spectral-residual-with-solver` recorded. Regenerating the
table belongs to `benchmark-reproducibility`, which sits above both this change
and `utils-encoding-cleanup`, the two changes beneath it that move a row.

**One loose end for the branches above.** `IN_FLIGHT` in
`tests/test_scenario_coverage.py` was introduced on this branch together with
this change's four exemption entries. All four are now gone, so on this branch
the constant is defined and unused. It is kept rather than removed because the
branches stacked above consume it for their own in-flight scenarios, and removing
it here would break them on rebase for no gain.

### Strengthening pass over the tests this change introduced

Each of the three anchors was re-read against what it actually proves, then
rewritten. All three are now fuzzed and all three are checked against deliberate
mutations of the thing they guard.

**The symmetry anchor was not testing the generator.** As first written it
rebuilt the band by hand from `FourierBasis.generate_coeff` and padded it, so it
anchored the arithmetic `2K + 1` rather than `mms_solution`. It would have passed
unchanged if the generator had stopped calling `widest_manufactured_band`
entirely — the change's headline claim, unguarded. It now calls `mms_solution`
and measures its returned field, with `resize_modes` replaced by the
repair-disabled variant through `unittest.mock.patch` rather than the
`monkeypatch` fixture, which Hypothesis refuses to reset between generated
inputs.

**A rounding floor was found while adding a value-space assertion.** The
coefficients are *exactly* Hermitian — `0.0` in all 210 cases swept over mode
counts 6–40 and seeds 0–5 — but the inverse transform's imaginary part is not
exactly zero, because the transform itself rounds in float32. Measured worst
`|imag| / |real|` is **9.550e-08** at modes 36, seed 3, against a float32 eps of
1.192e-07: the imaginary part sits at the rounding floor, not above it.
`MANUFACTURED_IMAGINARY_FLOOR = 1e-6` is set one order of magnitude above the
measurement, not tuned to pass.

**The stated-band anchor was checking a proxy.** It counted non-zero
coefficients at one mode count under `no_fuzz`. A count does not say the
non-zeros are the *right* ones. It now fuzzes mode count, seed and how far below
the default the stated cutoff sits, and asserts the support directly: every
coefficient at `|k| > K` is exactly zero, the kept count is `2K + 1`, and a band
narrower than the default actually changes the field rather than silently
matching it.

**The refusal anchor tested one overrunning band.** Seed was pinned and only
`widest + 1` was tried. It now fuzzes the seed and the overrun by 1–6 cutoffs,
and adds the assertion the whole derivation exists for: the *admitted* band's
continuous residual is inside `MANUFACTURED_RESIDUAL_TOLERANCE`, and the refused
pair's exceeds it by more than a factor of 100 while its own de-aliased residual
stays below `1e-10`.

Mutation check — each mutant replaces `widest_manufactured_band`, then the three
anchors run:

| mutant | caught by |
| --- | --- |
| pre-change count-based band, `max(1, m // 3)` | all 3 |
| one cutoff too wide, `m // 6 + 1`, breaching `2K <= m/3` | 2 of 3 |
| always-even count, `2 * (m // 6)` | all 3 |

The survivor in the middle row is the stated-band support anchor, and correctly
so: its scenario is about bands the *caller* states, and a wrong default is the
other two anchors' subject. No mutant survives all three.

`just check`: 306 passed, ruff/mypy/ty clean.
