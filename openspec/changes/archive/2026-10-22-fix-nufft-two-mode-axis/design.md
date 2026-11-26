## Context

See proposal.md for the defect. Approximate evaluation reaches torchkbnufft's `KbNufft` through one function, `nufft_evaluate`, called from `FourierBasis.evaluate` (dense work above the strategy's budget) and from `FourierBasis._raw_transform` (an inverse transform onto a large resolution). `KbNufft` defaults to six interpolation neighbours per axis and an oversampled grid of twice the mode count, and its table build indexes neighbours on that grid, so it fails whenever the grid is smaller than the neighbourhood: twice 1 and twice 2 are below 6, twice 3 is not.

Measured on the 1-D path against dense evaluation, 200 random points, relative to the largest exact value:

| modes | default grid | grid of at least 6 |
| --- | --- | --- |
| 1 | IndexError | 8.2e-6 |
| 2 | IndexError | 2.9e-4 |
| 3 | 5.7e-4 | 5.7e-4 (same grid) |
| 8 | 8.6e-4 | same grid |
| 16 | 8.1e-4 | same grid |

## Goals / Non-Goals

**Goals:**
- Every mode count the exact path accepts evaluates on the approximate path, within the bound the suite already holds (one twentieth of the largest exact value).
- No input that evaluated before evaluates differently.

**Non-Goals:**
- Tightening the approximation bound. The measured error is near 1e-3 against a bound of 5e-2; a tighter contract would cover every size and is a separate decision.
- Changing when a strategy chooses the approximate path.

## Decisions

**Raise the grid to the interpolation neighbourhood, not fall back to the exact path.** `nufft_evaluate` passes `grid_size` per axis as the larger of twice the mode count and the neighbourhood size. For axes of three or more modes that is exactly today's grid, so their values do not move by a bit. The alternative, sending any field with a one- or two-mode axis to the exact dense path, also removes the crash, but a field such as 1 x 400000 points over a 2-mode axis with a wide second axis is exactly the case the approximate path exists for, and the dense path would materialise the full basis matrix there. The neighbourhood size is stated once, as a module constant of 6 (torchkbnufft's default), and passed both as the operator's `numpoints` and as the grid's floor, so the two cannot drift apart. *Amended during apply:* the plan read it from the operator's signature; a spy test on the operator could not read a signature, and stating the value is the more explicit form.

**State the contract in a new capability.** No existing capability describes evaluation: `coefficient-encodings` covers conversions, resizing and band-limiting, and only says a conversion honours its strategy. `basis-evaluation` holds the two requirements; the conversion test in `coefficient-encodings` gains its 2-mode case back without a requirement change.

**Oracle and measurement.** The tests compare against exact dense evaluation, which shares no code with torchkbnufft (an independent oracle). Before the bound is asserted over the widened ranges, the worst relative error is measured over modes 1 to 48 on one axis and 1 to 12 on two and three axes, recorded in `measurements/`, and the existing 5e-2 bound is confirmed to sit above it.

**Recorded numbers.** The only behaviour that changes is the grid for an axis of one or two modes, where the operator raised before. A recorded result exists only for runs that finished, so none took that path, and the diff argument is sufficient; no benchmark rerun is needed.

## Risks / Trade-offs

- [torchkbnufft changes its default neighbourhood in a later release] → the library states its own, so its values do not follow the dependency's default; a test pins the operator's arguments, and the 1- and 2-mode pinned cases fail first if the table build changes again.
- [A grid of 6 for a 1-mode axis costs more than the dense path for tiny inputs] → negligible; the approximate path is only chosen above the memory budget, where the other axes dominate.
- [Mode count 1 is accepted by the exact path but is an unusual input] → it is pinned anyway, since the crash covered it.
