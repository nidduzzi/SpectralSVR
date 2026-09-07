## Execution order

```
1 the type ── 2 zero_coeff ──┬── 3 anchors ──┬── 5 verify
                             └── 4 notebooks ┘
```

Sections 3 and 4 both depend on 2 and on nothing else, so they may run in either
order. Nothing depends on a later section. The raw form reproduces today's
behaviour exactly, so the suite is green at the end of every section.

## 1. The band type

- [x] 1.1 Add a band type in `SpectralSVR/utils/__init__.py` with a cutoff form keeping every `|k| <= K`; verify it reports `2K + 1` modes and that its kept set is symmetric about the zero mode
- [x] 1.2 Add a whole-grid form; verify it keeps every bin of the array it is applied to, including the self-paired boundary bin on an even-sized grid, and that this is the case a cutoff cannot express — on 16 bins the cutoff forms give 1, 3, … 15 and never 16
- [x] 1.3 Add a raw-count form keeping exactly `m` modes, whose docstring states the result is not guaranteed to be conjugate-symmetric and names the cutoff form as the alternative when a real result is required; verify the wording states the limitation without promising a future change
- [x] 1.4 Verify no form of the type can express an even kept set that claims to be real: the cutoff form yields only odd sizes, the whole-grid form yields the grid size, and the raw form makes no claim
- [x] 1.5 Export the band forms and both aliases from the package root beside `zero_coeff` — five names, since the type is a closed union rather than one class; verify they import

## 2. `zero_coeff` takes a band

- [x] 2.1 Change `zero_coeff`'s parameter from `zeroed_modes: int | list[int]` to the band type, accepting one band or one per axis; verify a bare integer is rejected rather than silently interpreted
- [x] 2.2 Fix the empty-range crash: `zero_coeff(x, N)` raises `IndexError: index_fill_(): Expected dtype int64 for index` today for every grid size tried (16, 8, 7), because `torch.tensor(range(8, 8))` is `float32`. Verify the whole-grid form is a no-op returning a grid equal to its input, and that a band wider than the grid is handled rather than raising
- [x] 2.3 Verify the raw form reproduces the current behaviour exactly for every count that works today: for a fixed input, the result is `torch.equal` to what the pre-change function returns for the same count, over both parities and several grid sizes
- [x] 2.4 Verify the cutoff form preserves conjugate symmetry: a symmetric source band-limited by any cutoff reads Hermitian error `0`, where the equivalent even count reads `1.124` at keep 4, `0.557` at 6 and `0.633` at 8 today
- [x] 2.5 Verify per-axis bands still work on a multi-axis grid, including one axis stated by cutoff and another by raw count
- [x] 2.6 Update `zero_coeff`'s docstring: it currently describes the even-count limitation as unavoidable, which the cutoff form removes. Verify it describes what each band form gives rather than warning about a request that can no longer be made by accident

## 3. The two anchors

- [x] 3.1 Restate `test_zero_coeff_values` (`tests/basis/test_coeff_anchors.py:144`). It pins `zero_coeff(x, 4)` against a hand-written literal and is `oracle_analytic`; re-derive the expected array by hand from the bin layout for whichever band form the test now states, rather than recomputing it from the function under test. Verify the derivation was independent: the expected array is written before the new code is run against it and matches on the first run, not adjusted afterwards to agree. Verify it still discriminates by checking the restated anchor fails against a raw-count band of the same width, which is the distinction this change introduces
- [x] 3.2 Add a companion anchor for the cutoff form asserting conjugate symmetry is preserved, and one for the whole-grid form asserting the no-op; verify the symmetry anchor fails against a raw-count band of the same width, which is what makes it a discriminating test rather than a restatement
- [x] 3.3 Update `test_zero_coeff_zeros_high_modes` (`tests/utils/test_utils.py:342`), whose strategy fuzzes `keep` over `[1, 3]` and so includes the even 2; verify the marker still matches what the test anchors against
- [x] 3.4 Verify `tests/test_oracle_coverage.py` passes with the markers chosen

## 4. The notebooks

- [x] 4.1 Convert the four `zero_coeff` call sites — they sit two apiece in `notebooks/spectralsvr_inv.ipynb` and `notebooks/spectralsvr_hf_lf.ipynb`, while `notebooks/spectralsvr_2d ls_coeff.ipynb` and `notebooks/spectralsvr_wave2d.ipynb` import the name without ever calling it; for each call, record whether the original count was even or odd, since an even one was silently producing a complex field
- [x] 4.2 Choose each call's band form by what that notebook is doing rather than mechanically: a cutoff where the field is real and the intent is band-limiting, the raw form where the count was load-bearing for a downstream shape. Verify the choice is recorded per call site
- [x] 4.3 Verify each converted notebook still executes, or record why it cannot be run and what was checked instead — the notebooks may be outdated independently of this change

### What the four call sites turned into

The four call sites live in two notebooks, not four. `spectralsvr_2d ls_coeff.ipynb`
and `spectralsvr_wave2d.ipynb` import `zero_coeff` and never call it, so they
were left alone; `spectralsvr_inv.ipynb` and `spectralsvr_hf_lf.ipynb` carry two
calls each, in cell 5.

| notebook | count before | parity | chosen form |
| --- | --- | --- | --- |
| `spectralsvr_inv.ipynb` (both calls) | `modes = [100]` | even | `WavenumberCutoff(49)` |
| `spectralsvr_hf_lf.ipynb` (both calls) | `modes = [200]` | even | `WavenumberCutoff(99)` |

Both were even, so both were silently producing a grid that does not describe a
real field — and both plot `.real` of the result, so the discarded imaginary
part was never visible. The cutoff is `m // 2 - 1`: the widest band closed under
negation *inside* the set the count kept, so the kept modes are a subset of what
was kept before and the only bin dropped is the unpaired one that caused the
defect. The raw form was not chosen anywhere: at both sites the result is only
inverse-transformed and plotted, so no downstream shape depends on the count.

Neither notebook executes, for reasons that predate this change: both import
`skripsi_program` and call `FourierBasis.waveNumber`, `generateCoeff` and
`reduce_coeff`, none of which occur anywhere in `SpectralSVR/` today, so cell 0
fails on import. Checked instead: both files still parse as JSON, the converted
cells read correctly, and the band semantics they now rely on are the ones
`test_zero_coeff_cutoff_preserves_conjugate_symmetry` pins.

## 5. Verification

- [x] 5.1 Run `just check`; verify ruff, mypy and ty are clean and the suite passes
- [x] 5.2 Verify no reported number moved: `zero_coeff` has no callers inside `SpectralSVR/`, so regenerate the benchmark and confirm all four rows and the seeded numerical Burgers field are unchanged
- [x] 5.3 Verify the archived `coefficient-encodings` requirement now reads true: band-limiting through a real-guaranteeing band returns a conjugate-symmetric grid for every band that can be stated, so the operation no longer depends on documentation to warn about an even count
- [x] 5.4 Verify `mms_solution` is untouched — its band derivation belongs to the change stacked above this one, and touching it here would move the Burgers MMS row and break this change's claim to move nothing
- [x] 5.5 Verify no source file gained an explanatory prose comment; the band forms must read from their names, with the contracts confined to docstrings

### What 5.2 could and could not establish

Two of the four rows came back byte-identical (project antiderivative, Burgers
MMS). The other two moved, worst relative difference `2.512e-05` — which is the
run-to-run noise floor measured earlier in this stack (`2.51e-05`), not a change
in behaviour. An accidental head-against-head run served as the control: two
runs of *identical* code moved the Burgers MMS row too (`r2` `0.467641` vs
`0.467643`). So the benchmark table cannot carry a byte-identity claim at this
granularity, exactly as `dtype-seam` found.

What does carry it:

- The seeded numerical Burgers field is byte-identical between a clean worktree
  at `7df8bfe` and this tree: `sha256 0ad8fb33ce3237ccb641b1628ce19128668a591ede09f9b6b1d89d63b50340a9`,
  shape `(50, 100, 16)`, `complex64`.
- `zero_coeff` and the band type have no call sites anywhere in `SpectralSVR/`
  or `scripts/` — the only occurrences outside `utils/__init__.py` are the six
  re-export lines in `SpectralSVR/__init__.py`. No numerical path reaches them,
  so no row can move through this change.
- `SpectralSVR/problems/` has an empty diff, so `mms_solution` and its
  `band_limited = tuple(max(1, m // 3) for m in modes)` are untouched (5.4).

The base run required a separate environment in a detached worktree: a
`PYTHONPATH` override loses to the editable install, and a first attempt at the
base numbers silently ran the head library instead. Caught by printing
`zero_coeff`'s signature before benchmarking; that first attempt is the
head-against-head control quoted above.

### Where 2.3's comparison came from

`ExactModeCount` was checked against outputs captured from the pre-change
`zero_coeff` *before* the type was written: 52 `(grid size, count)` pairs over
grids `7, 8, 9, 12, 16` and counts `0..n+1`, all `torch.equal`. The 10 pairs
that raised `IndexError: index_fill_(): Expected dtype int64 for index` before
are now no-ops. That comparison ran against a captured baseline file and so
could not stay in the suite; `test_exact_count_keeps_the_lowest_frequencies_from_each_end`
carries the same behaviour forward as a hand-derived anchor.
