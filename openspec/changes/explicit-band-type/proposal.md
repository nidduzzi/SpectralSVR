# A band type whose invalid states cannot be constructed

## Why

`zero_coeff` (`SpectralSVR/utils/__init__.py:440`) band-limits a coefficient grid
in place. It carries three defects, all measured on the current branch.

**An even keep-count returns a grid that does not describe a real field.** On a
16-bin conjugate-symmetric source:

| keep | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| Hermitian error | 0 | 1.124 | 0 | 0.557 | 0 | 0.633 |

The cause is structural. Modes pair as `±k` around the lone mode `0`, so a kept
set whose inverse transform is real is odd-sized, and an even count necessarily
keeps one member of a pair without the other. `utils-encoding-cleanup` documented
this and deliberately left the behaviour alone, because every count-based repair
substitutes a different request for the one made: rounding to `m+1` or `m-1`
keeps a different number of modes than asked, and folding does not help at all —
measured `1.111`, `0.987`, `0.201`, no better than doing nothing, because
band-limiting keeps the array's length so the folded bin's partner is still
zeroed.

**Keeping everything crashes.** `zero_coeff(x, N)` at the grid size raises
`IndexError: index_fill_(): Expected dtype int64 for index` for every `N` tried
(16, 8, 7), as does any count above `N`. The zero range is empty,
`torch.tensor(range(8, 8))` is `float32`, and `index_fill_` demands `int64`. A
no-op raises.

**The parameter lies about itself.** It is named `zeroed_modes` and counts the
modes *kept*. `zero_coeff(x, 4)` keeps four.

## What Changes

- A band type is introduced with several constructors, so a caller names the
  intent rather than passing a bare integer whose meaning is ambiguous:
  - a **cutoff** form keeping every `|k| <= K`, yielding `2K + 1` modes, real by
    construction;
  - a **full-grid** form keeping the whole array including the self-paired
    boundary bin, also real;
  - a **raw-count** form keeping exactly `m` modes, carrying a documented
    contract that the result is *not* necessarily real.
- **BREAKING**: `zero_coeff` takes that type instead of an `int | list[int]`.
  Four notebooks and two tests call it and are updated.
- Keeping the whole grid stops raising and becomes the no-op it should be.
- The invalid state — an even count silently believed to be real — becomes
  unconstructible. Producing it now requires naming the raw form and accepting
  its stated contract.

The raw form is deliberate, not a loophole. A cutoff-only type would be too
strong for two measured reasons. It cannot express the full grid on an
even-sized array: on 16 bins the valid symmetric kept sets are 1, 3, … 15 *and
also 16*, because at the grid's own edge the bin at `-N/2` is self-paired and
keeping everything is real, so the "always odd" rule holds strictly inside the
grid and breaks at its boundary. And non-Hermitian grids are legitimate input —
`enforce_hermitian` is applied in exactly one place, `FourierBasis.generate_coeff:363`,
and model predictions are never symmetrised — so `zero_coeff` receives grids
where a real result is not the premise, and an even count there is ordinary
truncation rather than a defect. Making oddness a type invariant would impose a
realness requirement on callers who never asked for one.

## Capabilities

### New Capabilities
- `spectral-band-selection`: how a caller states which modes an operation keeps,
  what each form of that statement guarantees about the result being real, and
  which forms can be constructed at all.

### Modified Capabilities
- `coefficient-encodings`: the requirement "An operation that cannot return a
  real field says so" was written for `zero_coeff` when no repair existed. Once
  the type lands the operation *can* return a real field for every band the safe
  constructors produce, so that requirement's band-limiting scenario is restated
  rather than dropped — the raw form still cannot promise realness and must
  still say so.

## Impact

- `SpectralSVR/utils/__init__.py`: the band type and `zero_coeff`'s signature and
  body.
- `SpectralSVR/__init__.py`: export the type beside `zero_coeff`.
- `tests/basis/test_coeff_anchors.py:144`: `test_zero_coeff_values` pins an even
  count of 4 by hand-written literal and is restated in terms of a band.
- `tests/utils/test_utils.py:342`: `test_zero_coeff_zeros_high_modes` fuzzes
  `keep` over `[1, 3]`, which includes the even 2.
- `notebooks/`: `spectralsvr_inv.ipynb`, `spectralsvr_2d ls_coeff.ipynb`,
  `spectralsvr_hf_lf.ipynb`, `spectralsvr_wave2d.ipynb` all call `zero_coeff`.
- **No reported number moves.** `zero_coeff` has no callers inside
  `SpectralSVR/` — only notebooks and tests — so no benchmark row and no seeded
  field can change through it.

`mms_solution`'s band derivation in `problems/burgers.py`, which computes
`band_limited = tuple(max(1, m // 3) for m in modes)` and can therefore be even,
is deliberately **not** in scope. It moves the Burgers MMS benchmark row, and
this change moves nothing, so it is a separate change stacked above this one.
