# The manufactured field states its band instead of computing one

## Why

`Burgers.mms_solution` (`SpectralSVR/problems/burgers.py:337`) band-limits its
manufactured field by computing `band_limited = tuple(max(1, m // 3) for m in
modes)`, generating an array that size, and padding it to full width. The band is
expressed as a *grid size* — make a small array, then pad — and `m // 3` is even
for most mode counts, so the small array has an unpaired top coefficient and the
field it describes is not real.

Measured with `resize_modes` left untouched, varying only the band's parity:

| mode count | 8 | 12 | 16 | 32 |
| --- | --- | --- | --- | --- |
| band `m // 3` | 2 (even) | 4 (even) | 5 (odd) | 10 (even) |
| Hermitian error | 7.859e-01 | 2.957e+00 | 0.000e+00 | 7.392e+00 |
| with the band forced odd | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Mode count 16 is clean before any change because `16 // 3 = 5` is the only odd
band among them. That is the isolation control: parity is the whole of it.

`utils-encoding-cleanup` repaired the consequence, not the cause. Its
`resize_modes` boundary rule makes padding preserve conjugate symmetry, so
`u.coeff` is symmetric today. That repair is correct and stays — grids arrive
from user data, saved coefficients and model predictions where no type protected
them, and an even-sized source is permanently legitimate because FFT sizes are
usually even. But the manufactured field is generated here, under this library's
control, and should never be malformed in the first place. The defensive fix
downstream and the constructive fix upstream are complements, not alternatives.

## What Changes

- `mms_solution` states its band through the cutoff form of the band type from
  `explicit-band-type`, keeping `|k| <= K`, so no derivation can produce an
  unpaired top coefficient.
- Expressed as a cutoff the de-aliasing threshold becomes exact rather than
  approximate. Multiplication doubles the highest frequency, so the product
  reaches exactly `2K` while the Orszag 2/3 mask keeps `m / 3`, giving
  `2K <= m / 3`. The derived default is the largest cutoff satisfying it.
- `Burgers.generate` gains an optional band argument, threaded to
  `mms_solution` and defaulting to the derived cutoff. It sits beside `nu` and
  `solver` as a Burgers-specific parameter, not on the `Problem.generate`
  contract — nothing else has a band, because only this branch forms a nonlinear
  product needing de-aliasing headroom.
- **A stated band that overruns the threshold raises.** The knob can only narrow
  the manufactured problem. A wider band yields a pair that is self-consistent,
  looks like a manufactured solution, and does not solve the continuous equation
  — measured during `align-spectral-residual-with-solver` at `m = 16`: `m//2`
  leaves `1.50e-1`, `m//2 - 1` leaves `7.56e-2`, `m//3` reaches `3.64e-7`. That
  is the defect class this chain of changes exists to remove, so it is refused
  rather than warned about.
- The recorded manufactured-residual values move, because the band's width moves.

The band is expected to get *wider*, not narrower, which the tasks verify rather
than assume. A count-based band `B` spans modes `-(B//2) … +((B-1)//2)`, so its
highest frequency is `B//2` and one slot is spent on the asymmetry. A cutoff with
the same highest frequency keeps `2K + 1` modes symmetric about zero. At 8 modes
the current band is 2 wide with a highest frequency of 1; a cutoff of 1 has the
same highest frequency and keeps 3 modes — same bandwidth, one more mode, and
conjugate-symmetric.

## Capabilities

### Modified Capabilities
- `manufactured-solutions`: the field's band becomes part of what the capability
  guarantees — that the manufactured field is real by construction rather than by
  downstream repair, and that a band which would break the continuous-equation
  guarantee is refused rather than accepted.

## Impact

- `SpectralSVR/problems/burgers.py`: `mms_solution`'s band derivation and
  signature; the manufactured branch of `Burgers.generate` at `:90` threads the
  new optional argument; the numerical branch rejects it, mirroring how the
  manufactured branch already rejects `solver`.
- `tests/problems/test_problems.py`: `MANUFACTURED_CONTINUOUS_RESIDUAL`'s recorded
  values move with the band's width; the `mms_solution` call sites at `:324`,
  `:744` and `:762` are unaffected unless they state a band.
- **The Burgers MMS benchmark row moves**, because `u.coeff` is the regression
  target and reaches the model through `to_real_coeff`. Regenerating the table is
  **not** a task here — `benchmark-reproducibility` owns it and must sit above
  both this change and `utils-encoding-cleanup`'s resize repair, the two changes
  that move a row beneath it.

Not in scope: `resize_modes`, which keeps its boundary rule — this change removes
the need for it in one caller, not in general. `zero_coeff` and the band type
itself, which belong to `explicit-band-type`. The value of the `2/3` fraction,
which is not being retuned; only the rounding within it. And deliberately
aliased datasets for robustness study: that needs the archived spec's
continuous-versus-discrete overrun scenarios reconciled first, and refusing an
overrunning band here keeps that conflict unreachable rather than inheriting it.
