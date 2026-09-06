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

### The model boundary refuses in one direction and reports in the other

At `model/SpectralSVR.py:273` the interleaved real values are handed to the regressor. The regressor's precision is set when it is constructed, so by the time the boundary is reached, disagreement is a construction error, not something to paper over.

The two directions are not the same error. Values *wider* than the regressor lose information a cast cannot give back, so that is refused, naming both precisions. A regressor *wider* than the values loses nothing — it only pays double-precision cost for single-precision data, a 16.9× matmul penalty here for no accuracy gain. That is a waste, not a loss, and a warning is proportionate to a cost in a way it is not to the destruction of a result.

*Discovered during implementation.* The symmetric rule the tasks first stated breaks two anchors that are right to do what they do: `test_forward_matches_closed_form_antiderivative` and `test_inverse_recovers_known_input` pair a `complex64` basis with `LSSVR(dtype=torch.float64)`, because their closed-form comparison needs the precision in the *solve* while the basis has no reason to leave the default. The spec already permitted this — it forbids silent *truncation*, and widening truncates nothing.

The guard also applies only to values carrying a basis's coefficient precision through the bridge. Real inputs a caller supplies at a precision of their own have always been cast, and the basis's configuration makes no claim about them; guarding those broke the float64 query grids in `tests/operators/test_operators.py`, which is how the distinction surfaced.

*Alternative rejected:* upcast the regressor to match. Hides a configuration mistake and can silently make a run far slower than intended, which on this hardware is a 16.9× surprise.

*Alternative rejected:* keep truncating but warn. A warning in a numerical library is a result nobody checks; the current silence is only worse by degree. That rejection stands, and is why the lossy direction raises. It does not extend to the widening direction, where there is no wrong result to miss — only a bill to notice.

*Note on convention.* The library has exactly one `logger.warning`, in the plotter. Adding a second is a deliberate departure, taken because the alternative for a merely wasteful pairing is either silence or a refusal that would reject working configurations.

### The bridge is hardened, not redesigned

`to_complex_coeff` gains `.contiguous()` before the view, and an optional target dtype. Both are additive: existing callers pass contiguous input and rely on inference, and keep working unchanged. The `.contiguous()` is one copy on a path that already allocates.

Both edits land on a function whose guarantees are stated but untested. `coefficient-encodings` binds *every* conversion to accept any number of mode axes and to preserve device and precision, and only the polar pair carries tests for it; the interleaved-real pair satisfies the requirement today with nothing defending it. The anchor therefore comes before the edit in the task order, so a regression introduced by the hardening is caught by a test that was already passing rather than by one written afterwards to match.

*Alternative rejected:* leave the bridge undefended and rely on the round-trip test in group 5. That test asserts value equality at both precisions and would not notice a lost device or a wrongly inferred target precision, which are exactly what the two edits could break.

### Verification is byte-identity plus a precision round-trip

The change's own proof is that nothing moved: the seeded numerical Burgers field and all four benchmark rows must be unchanged, since the default is untouched. The new capability is proved separately by constructing a double-precision basis and showing the precision survives to the regressor.

*The benchmark cannot serve as the byte-identity check, and this was measured rather than assumed.* Six consecutive runs of identical code in the same tree vary by up to `2.51e-05` relative, against a base-vs-head difference of `8.91e-06` for this change — the harness moves further against itself than this change moves it. Dataset generation is bit-identical and an isolated fit-and-predict is bit-identical within and across processes, so the variation is neither seeding nor the model; pinning `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 cuts it to `6.15e-07`, which points at multi-threaded reductions in the solve. That is a property of the harness, not of this change, and `benchmark-reproducibility` is the change that addresses it. The seeded field hash is what binds here.

A third anchor covers the bridge itself, built from a list of `(forward, inverse)` pairs rather than one test per function. The requirement it enforces is quantified over *every* conversion, so the test is quantified the same way: a conversion added to the package later is either covered by the registry or fails the anchor for being absent from it. Writing four near-identical tests would satisfy the requirement today and silently stop satisfying it the moment a fifth conversion appears.

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
