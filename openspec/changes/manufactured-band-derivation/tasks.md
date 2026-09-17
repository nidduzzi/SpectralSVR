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

- [ ] 2.1 Add an optional band parameter to `mms_solution`, defaulting to the derived cutoff; verify every existing call site is unaffected
- [ ] 2.2 Thread it through `Burgers.generate` as an explicit named parameter beside `nu` and `solver` — **not** through `**kwargs`, which `Burgers.generate` never reads (`ty` reports it unused at `burgers.py:50`) and would silently discard it. Verify the parameter is explicit rather than absorbed: a band stated through `Burgers.generate` reaches `mms_solution` and changes the generated field's width against the default, while the same value passed under an unrecognised keyword leaves the width at the default — the second case is the silent discard this task exists to prevent
- [ ] 2.3 Validate a stated band against `2K <= m / 3` and raise when it overruns, naming both the stated bandwidth and the widest the mode count admits; verify the message contains both numbers
- [ ] 2.4 Verify the refusal is justified by measurement rather than caution: a band at `m//2` leaves a continuous residual of `1.50e-1` and `m//2 - 1` leaves `7.56e-2`, against `3.64e-7` inside the threshold. Record that an overrunning pair is still self-consistent, which is why refusing at generation is the only place it can be caught
- [ ] 2.5 Make the numerical branch of `Burgers.generate` reject a band, mirroring how the manufactured branch already rejects `solver` at `burgers.py:84`; verify the error says why it cannot act rather than ignoring it
- [ ] 2.6 Verify a band that is structurally impossible — not positive, or wider than the grid — is refused separately from one that merely overruns the threshold, since they are different errors
- [ ] 2.7 Run `just check`; verify clean

## 3. Anchors

- [ ] 3.1 Add an anchor asserting the generated field is conjugate-symmetric at every supported mode count with the resize boundary rule disabled; verify it fails against the pre-change derivation, which is what makes it a test of this change rather than of the one beneath it
- [ ] 3.2 Add an anchor asserting an overrunning stated band raises, and that the widest admissible band does not; verify the boundary between them is exactly `2K <= m / 3` rather than one off it
- [ ] 3.3 Update `MANUFACTURED_CONTINUOUS_RESIDUAL` in `tests/problems/test_problems.py` with the re-measured values; verify they are recorded from measurement and that no tolerance was widened to accommodate them
- [ ] 3.4 Verify the analytic positive control and the corrupted-forcing negative control both still behave — the control pair is independent of the band, so a change in either indicates this change reached further than intended
- [ ] 3.5 Verify `tests/test_oracle_coverage.py` passes with the markers chosen

## 4. Verification

- [ ] 4.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 4.2 Verify `resize_modes` is untouched: this change removes the need for its boundary rule in one caller, not in general, and the anchors added by `utils-encoding-cleanup` must still pass unchanged
- [ ] 4.3 Verify `Antiderivative` is untouched and gained no band argument — it is linear, forms no product, and needs no de-aliasing headroom
- [ ] 4.4 Record that the Burgers MMS row moves and is regenerated by `benchmark-reproducibility`, which sits above this change; do not regenerate the table here
- [ ] 4.5 Verify no source file gained an explanatory prose comment; the derivation must read from the band form's name and the validation's error text
