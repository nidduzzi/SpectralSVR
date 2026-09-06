## Context

See proposal.md — Why for the three defects and their measurements. The
design-relevant code:

```
zero_coeff                      utils/__init__.py:440
  keeps range((m-1)//2 + 1, N - m//2)      <- empty when m >= N, and
                                              torch.tensor(empty) is float32
dealias_mask                    basis/FourierBasis.py:293
  k.abs() <= fraction * (modes / 2)        <- already a cutoff, not a count
enforce_hermitian               basis/FourierBasis.py:363 (sole call site)
```

| fact | consequence |
| --- | --- |
| `enforce_hermitian` is applied only inside `generate_coeff`; predictions are never symmetrised | non-conjugate-symmetric grids are legitimate input, so truncating one is not a defect |
| on 16 bins the valid real kept sizes are 1, 3, … 15 **and 16** | "always odd" holds strictly inside the grid and breaks at its boundary |
| `zero_coeff` has no callers inside `SpectralSVR/` | no seeded field or benchmark row can move through this change |
| `dealias_mask` already expresses a band as a cutoff on `\|k\|` | the correct concept exists in the codebase; the count-based sites are the ones that drifted |

## Goals / Non-Goals

**Goals:**

- Make the request that cannot be satisfied — an even kept set believed to be
  real — impossible to state, rather than caught or documented.
- Keep every request that *is* satisfiable statable, including the whole grid and
  the truncation of a grid that was never conjugate-symmetric.
- Make the call site say which modes it keeps.

**Non-Goals:**

- `mms_solution`'s band derivation. It moves the Burgers MMS benchmark row and
  this change moves nothing; separating them keeps that property provable.
- Changing what `zero_coeff` computes for any band that is expressible today and
  still expressible after. The raw form must reproduce current behaviour
  exactly.
- Introducing a band anywhere a *grid size* is meant. `resize_modes` takes a
  target mode count, which is storage, not signal; conflating the two is the
  defect this change exists to remove and must not be re-introduced in the other
  direction.

## Decisions

### The type carries several constructors rather than one cutoff

The tempting design is a single cutoff `K` keeping `|k| <= K`, giving `2K + 1`
modes — odd by construction, so an even band cannot exist. It is too strong.

It cannot express the whole grid on an even-sized array. On 16 bins the bin at
`-8` is its own conjugate partner, so keeping all 16 is real, and `2K + 1` yields
only 1, 3, … 15. And it forbids truncating a grid that is not conjugate-symmetric
at all, which a prediction is, where an even count is ordinary truncation rather
than a defect.

So the type offers a cutoff form, a whole-grid form, and a raw-count form whose
contract states it does not promise a real result. The invalid state is still
unconstructible: an even count *believed to be real* cannot be expressed, because
reaching an even count now means naming the raw form and taking its contract with
it.

*Alternative rejected — cutoff only.* Strongest invariant, and it makes two
legitimate operations unreachable. A caller wanting either would drop to raw
tensor slicing, which is worse than a named form with a stated contract.

*Alternative rejected — keep the integer and validate it.* Rejecting an even
count at run time catches the defect but leaves it expressible, and gives a
caller with a non-symmetric grid no way to say what they mean.

### The raw form is a peer, not a deprecated path

It is not a migration shim. Truncating a non-symmetric grid is a real operation
with no better spelling, and the library has a genuine source of such grids.
Marking it deprecated would push callers toward either a false realness claim or
raw slicing.

### `dealias_mask` is left alone for now

It already computes a cutoff, so it is the precedent rather than a defect site,
and adopting the type there is not free: its `fraction` is a *ratio of the grid*
(`2/3` by Orszag) rather than an absolute cutoff, so it would need a form that
resolves against a grid size at use. That is a coherent extension and it is not
needed to fix anything. Recorded as an open question rather than done quietly.

### The crash fix rides along because the whole-grid form requires it

`zero_coeff(x, N)` raises today. The whole-grid form is exactly that call, so it
cannot be delivered without fixing it. It is not separate scope.

### Both `zero_coeff` anchors are restated, not regenerated

`test_zero_coeff_values` pins a literal spectrum by hand and is `oracle_analytic`.
Its band moves from an even count of 4 to a stated band, and the expected array
is re-derived by hand from the bin layout. Recomputing it from the new code would
convert it into a test that asserts the code agrees with itself.

## Risks / Trade-offs

- **BREAKING for four notebooks.** → They are the only callers outside tests, and
  each call is a single argument. The raw form reproduces today's behaviour
  exactly, so a mechanical conversion is available where the intent is unclear
  and a cutoff where it is not.

- **Two ways to say something similar.** A reader may not know which form to
  reach for. → The forms are named for what they guarantee, and the raw form's
  contract names the alternative. This is the cost of not over-restricting, taken
  deliberately.

- **The escape hatch can be misused.** A caller wanting a real result could name
  the raw form and get a complex one. → It cannot be reached by accident from an
  ambiguous integer, which is the failure mode today; reaching it now is a
  choice, and the contract is stated at the point of that choice.

- **A band type invites use where a grid size is meant.** → Called out as a
  Non-Goal; `resize_modes` keeps its mode-count argument.

## Migration Plan

The type first, with the whole-grid form's crash fix, then `zero_coeff`, then the
two anchors, then the notebooks. Each step leaves the suite green: the raw form
reproduces current behaviour, so the anchors can be restated before or after the
notebooks without a window where either is broken.

Rollback is a revert. No persisted state. The only externally visible change is
`zero_coeff`'s signature.

## Open Questions

- Whether `dealias_mask` should express its band through the type, given it
  already computes a cutoff. It needs a grid-relative form to do so, which
  nothing else currently needs. Safely deferrable: it changes neither this
  change's approach nor its tasks.
