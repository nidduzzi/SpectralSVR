## Context

See proposal.md - Why for the gap and the measurements. The design-relevant surface:

```
basis side                         model side                    bridge
  Basis.coeff_dtype  __base.py:89    MultiRegression.dtype :43     to_real_coeff
    declared abstract                  float32, 4 cast sites       to_complex_coeff
  FourierBasis.coeff_dtype :29       LSSVR.dtype        :140        (already sound)
    CLASS CONSTANT complex64           float32
  generate_empty     :342  literal   SpectralSVR.py     :273
  generate_coeff     :360  literal     to_real_coeff(...).to(device)   <- no dtype
```

Facts established by probing, which the design leans on:

| fact | consequence |
| --- | --- |
| the bridge is bit-exact and transparent at both precisions, 2-D and 3-D | no redesign there, only hardening |
| `torchkbnufft` accepts complex128 | the NUFFT evaluation path needs no exclusion |
| `cdist` rejects complex | irrelevant — the kernel path sees only real interleaved data |
| matmul 2048² costs 16.9× more at double on this GPU | the default must not change |
| `model/SpectralSVR.py:273` passes device but not dtype | the boundary is where precision is actually lost |

## Goals / Non-Goals

**Goals:**

- One selectable precision that determines both sides, defaulting to today's behaviour.
- The model boundary either preserves precision or refuses, never truncates silently.
- The bridge accepts non-contiguous input and can be told a target precision.

**Non-Goals:**

- Changing the default. Every seeded field and benchmark row must stay identical, and the throughput measurement is the reason.
- Making the DeepONet baseline double precision. Its float32 casts are deepxde's requirement, not ours.
- Per-axis or mixed precision. One precision per basis; a mixed scheme has no motivating use here.
- Reworking `StandardScaler`, which already handles both precisions.

## Decisions

### The precision lives on `SpectralConfig`, expressed as the complex dtype

`SpectralConfig` is a frozen dataclass already holding domain, sampling and strategy, documented as the coefficient-independent configuration, with a `replace`-based `with_domain`. Precision is the same kind of knob: it does not depend on any particular coefficients. It joins that object with a matching `with_precision`.

The stored value is the complex dtype, with the real dtype derived from it, rather than the reverse. The basis's own storage is complex; the real form is what the bridge produces on the way to the model, so complex is the primary and real is the consequence.

*Alternative rejected:* an enum such as `single | double`. Reads well but adds a mapping to maintain, and every consumer ultimately wants a `torch.dtype`.

*Alternative rejected:* leave `coeff_dtype` as a class attribute and add a per-instance override. Two sources of truth for the same value, which is the situation being fixed.

### The class attribute becomes the default for the config, not a second source

`FourierBasis.coeff_dtype` stays readable, because `model/SpectralSVR.py` and `problems/burgers.py` consult it, but it resolves through the instance's configuration rather than being a constant. A basis used purely as a template, with no coefficients, still answers — the config exists independently of coefficients, which is the reason that object was carved out.

### The model boundary refuses rather than reconciles

At `model/SpectralSVR.py:273` the interleaved real values are handed to the regressor. The regressor's precision is set when it is constructed, so by the time the boundary is reached, disagreement is a construction error, not something to paper over. The boundary therefore checks and raises, naming both precisions.

*Alternative rejected:* upcast the regressor to match. Hides a configuration mistake and can silently make a run far slower than intended, which on this hardware is a 16.9× surprise.

*Alternative rejected:* keep truncating but warn. A warning in a numerical library is a result nobody checks; the current silence is only worse by degree.

### The bridge is hardened, not redesigned

`to_complex_coeff` gains `.contiguous()` before the view, and an optional target dtype. Both are additive: existing callers pass contiguous input and rely on inference, and keep working unchanged. The `.contiguous()` is one copy on a path that already allocates.

### Verification is byte-identity plus a precision round-trip

The change's own proof is that nothing moved: the seeded numerical Burgers field and all four benchmark rows must be unchanged, since the default is untouched. The new capability is proved separately by constructing a double-precision basis and showing the precision survives to the regressor.

## Risks / Trade-offs

- **Threading precision through the model constructors touches several signatures.** `MultiRegression`, `FNN` and `LSSVR` all carry their own default. → They keep their defaults; only the resolution changes, so existing call sites are unaffected.

- **A raising boundary turns a previously silent path into an error.** Anyone who was unknowingly relying on the truncation gets a failure. → That truncation was losing precision without telling them; the error names both sides so the fix is obvious. Nothing in the repo currently pairs mismatched precisions.

- **Double precision on this GPU is dramatically slower for the kernel and solve.** A user selecting it for a large benchmark may be surprised. → The cost is stated in the documentation at the point of choice, which the capability requires.

- **`torchmetrics` accepts complex128 and returns a meaningless complex value.** If a future caller routes metrics before the real conversion, it fails silently. → Out of scope, but worth a note where the metric suite is documented; the current pipeline always converts first.

## Migration Plan

1. Add the precision field to `SpectralConfig` with the current default, plus its `with_` helper; nothing else changes yet, and the suite must be untouched.
2. Route `FourierBasis.coeff_dtype`, `generate_empty` and `generate_coeff` through it; verify byte-identity of seeded output.
3. Thread the real-side precision from the same decision; add the boundary check.
4. Harden `to_complex_coeff` and route the ETDRK4 fallback and the benchmark's GRF sampler.
5. Add the double-precision round-trip test and the mismatch test.
6. Confirm all four benchmark rows and the seeded field are unchanged.

Rollback is a revert. No persisted state, no data format change.

## Open Questions

- Whether the metric suite should reject complex input outright rather than returning a complex number. Deferrable: the pipeline converts before metrics, so it cannot currently arise.
