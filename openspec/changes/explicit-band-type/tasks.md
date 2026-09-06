## Execution order

```
1 the type ── 2 zero_coeff ──┬── 3 anchors ──┬── 5 verify
                             └── 4 notebooks ┘
```

Sections 3 and 4 both depend on 2 and on nothing else, so they may run in either
order. Nothing depends on a later section. The raw form reproduces today's
behaviour exactly, so the suite is green at the end of every section.

## 1. The band type

- [ ] 1.1 Add a band type in `SpectralSVR/utils/__init__.py` with a cutoff form keeping every `|k| <= K`; verify it reports `2K + 1` modes and that its kept set is symmetric about the zero mode
- [ ] 1.2 Add a whole-grid form; verify it keeps every bin of the array it is applied to, including the self-paired boundary bin on an even-sized grid, and that this is the case a cutoff cannot express — on 16 bins the cutoff forms give 1, 3, … 15 and never 16
- [ ] 1.3 Add a raw-count form keeping exactly `m` modes, whose docstring states the result is not guaranteed to be conjugate-symmetric and names the cutoff form as the alternative when a real result is required; verify the wording states the limitation without promising a future change
- [ ] 1.4 Verify no form of the type can express an even kept set that claims to be real: the cutoff form yields only odd sizes, the whole-grid form yields the grid size, and the raw form makes no claim
- [ ] 1.5 Export the type from the package root beside `zero_coeff`; verify it imports

## 2. `zero_coeff` takes a band

- [ ] 2.1 Change `zero_coeff`'s parameter from `zeroed_modes: int | list[int]` to the band type, accepting one band or one per axis; verify a bare integer is rejected rather than silently interpreted
- [ ] 2.2 Fix the empty-range crash: `zero_coeff(x, N)` raises `IndexError: index_fill_(): Expected dtype int64 for index` today for every grid size tried (16, 8, 7), because `torch.tensor(range(8, 8))` is `float32`. Verify the whole-grid form is a no-op returning a grid equal to its input, and that a band wider than the grid is handled rather than raising
- [ ] 2.3 Verify the raw form reproduces the current behaviour exactly for every count that works today: for a fixed input, the result is `torch.equal` to what the pre-change function returns for the same count, over both parities and several grid sizes
- [ ] 2.4 Verify the cutoff form preserves conjugate symmetry: a symmetric source band-limited by any cutoff reads Hermitian error `0`, where the equivalent even count reads `1.124` at keep 4, `0.557` at 6 and `0.633` at 8 today
- [ ] 2.5 Verify per-axis bands still work on a multi-axis grid, including one axis stated by cutoff and another by raw count
- [ ] 2.6 Update `zero_coeff`'s docstring: it currently describes the even-count limitation as unavoidable, which the cutoff form removes. Verify it describes what each band form gives rather than warning about a request that can no longer be made by accident

## 3. The two anchors

- [ ] 3.1 Restate `test_zero_coeff_values` (`tests/basis/test_coeff_anchors.py:144`). It pins `zero_coeff(x, 4)` against a hand-written literal and is `oracle_analytic`; re-derive the expected array by hand from the bin layout for whichever band form the test now states, rather than recomputing it from the function under test. Verify the derivation was independent: the expected array is written before the new code is run against it and matches on the first run, not adjusted afterwards to agree. Verify it still discriminates by checking the restated anchor fails against a raw-count band of the same width, which is the distinction this change introduces
- [ ] 3.2 Add a companion anchor for the cutoff form asserting conjugate symmetry is preserved, and one for the whole-grid form asserting the no-op; verify the symmetry anchor fails against a raw-count band of the same width, which is what makes it a discriminating test rather than a restatement
- [ ] 3.3 Update `test_zero_coeff_zeros_high_modes` (`tests/utils/test_utils.py:342`), whose strategy fuzzes `keep` over `[1, 3]` and so includes the even 2; verify the marker still matches what the test anchors against
- [ ] 3.4 Verify `tests/test_oracle_coverage.py` passes with the markers chosen

## 4. The notebooks

- [ ] 4.1 Convert the four `zero_coeff` call sites in `notebooks/spectralsvr_inv.ipynb`, `notebooks/spectralsvr_2d ls_coeff.ipynb`, `notebooks/spectralsvr_hf_lf.ipynb` and `notebooks/spectralsvr_wave2d.ipynb`; for each, record whether the original count was even or odd, since an even one was silently producing a complex field
- [ ] 4.2 Choose each call's band form by what that notebook is doing rather than mechanically: a cutoff where the field is real and the intent is band-limiting, the raw form where the count was load-bearing for a downstream shape. Verify the choice is recorded per call site
- [ ] 4.3 Verify each converted notebook still executes, or record why it cannot be run and what was checked instead — the notebooks may be outdated independently of this change

## 5. Verification

- [ ] 5.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [ ] 5.2 Verify no reported number moved: `zero_coeff` has no callers inside `SpectralSVR/`, so regenerate the benchmark and confirm all four rows and the seeded numerical Burgers field are unchanged
- [ ] 5.3 Verify the archived `coefficient-encodings` requirement now reads true: band-limiting through a real-guaranteeing band returns a conjugate-symmetric grid for every band that can be stated, so the operation no longer depends on documentation to warn about an even count
- [ ] 5.4 Verify `mms_solution` is untouched — its band derivation belongs to the change stacked above this one, and touching it here would move the Burgers MMS row and break this change's claim to move nothing
- [ ] 5.5 Verify no source file gained an explanatory prose comment; the band forms must read from their names, with the contracts confined to docstrings
