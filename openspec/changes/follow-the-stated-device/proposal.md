## Why

The library overrides the device the user asked for:

```
global default cpu   -> resolve_device() returns cuda:0
global default cuda  -> resolve_device() returns cuda:0
```

Pin `cpu` and SpectralSVR still works on `cuda:0`. An explicit argument is
honoured, but the common case — say nothing, or say it once for the whole
process the way a torch user does — is answered by the library's own preference
instead of the user's.

Three components answer "what device?" and two are wrong:

| | component | answer today |
| --- | --- | --- |
| ✗ | `resolve_device(None)` | always `cuda:0` when CUDA is present, whatever the process says |
| ✗ | `MultiRegression` | hardcoded `cpu`, whatever anyone says |
| ✓ | bare tensor factories | follow the process default |

The placement this produces is inverted from what anyone would choose: the heavy
solve runs on the host while the bulk data goes to the accelerator. Generating
one dataset on the accelerator peaks near **1 GB** of VRAM to hold roughly
**320 MB** of coefficients, buying nothing, while `LSSVR.fit` gives up
**3–4×** at `(2000, 512)` and the ETDRK4 solve gives up **10–20×** at
`(500, 64, 800)`.

That last figure is unreachable today at any price: `Problem.generate` has no
device parameter at all, so a caller cannot ask for the accelerator even when it
is worth an order of magnitude.

Nothing caught this because the test harness is insulated from it. An autouse
fixture in `tests/conftest.py` snapshots and restores the process default around
every test, and its own docstring names the cause: "importing deepxde sets the
default device to cuda, which would otherwise leak into every later test". The
condition was identified and neutralised where tests run, and never where the
library or the benchmark run.

## What Changes

- `resolve_device(None)` returns the process default. An explicit argument still
  wins, unchanged.
- **New** `accelerator()`, wrapping `torch.accelerator.current_accelerator()` and
  falling back to the default when none is present. Device-agnostic, so MPS and
  XPU work — today's hardcoded `cuda:0` silently means no acceleration on a Mac.
- Every entry point that places work gains a device argument, so a caller can
  direct one stage without directing the process: the problem generators on the
  base and both subclasses, the basis generation entry points, and the model's
  inverse solve. This is what makes the order-of-magnitude gain reachable.
- `MultiRegression` drops its hardcoded `cpu` and routes through the seam; the
  stale comment acknowledging it goes with it.
- The model stops building its own random source on the host regardless of where
  it is working. `inverse` and `inverse_coeff` construct one with a fixed seed
  and no device today, so under a stated accelerator the library would trip the
  refusal below against a source the caller never supplied.
- The benchmark script states the device it runs on rather than inheriting
  whatever the last import left behind.
- A disagreeing generator is refused with both devices named. Today torch raises
  `Expected a 'cuda' device type for generator but found 'cpu'`, which does not
  say that the generator is the thing to move.

Measured, so the cost is known rather than hoped for: with the seam following the
default, the suite is **324 passed, 1 failed** — and the one failure is the test
asserting the behaviour being removed. The benchmark then runs end to end with
all four datasets and both operators, a configuration that crashes today. Metrics
are identical to about six significant figures; `r2` moves in its last digit.

Timings in this document are directions and orders of magnitude, not reproducible
constants. Host timings vary by a factor of two to three between runs on an
otherwise idle machine — a median can be six times the best of the same seven
runs — while accelerator timings repeat to a decimal place. Re-measuring will not
reproduce a figure; it should reproduce a winner and a rough ratio.

## Capabilities

### New Capabilities
- `device-selection`: who decides where work is placed, in what order of
  precedence, and what happens when a caller's random source disagrees with that
  decision.

### Modified Capabilities
<!-- None. `coefficient-precision` is the structural analogue — it states that
     precision is configuration rather than a constant — and this change follows
     its shape without altering it. The asymmetry between the two is a design
     decision, recorded in design.md. -->

## Impact

- `SpectralSVR/utils/__init__.py`: `resolve_device`, plus `accelerator()`.
- `SpectralSVR/problems/`: the base `generate` signature and both subclasses;
  `Burgers` stops calling `resolve_device()` bare.
- `SpectralSVR/basis/`: the generation entry points on the abstract base and on
  `FourierBasis`, which place work and accept no device today.
- `SpectralSVR/model/__base.py` and the stale comment in `LSSVR.py`;
  `SpectralSVR/model/SpectralSVR.py` for the inverse solve and the random source
  it builds for itself.
- `scripts/run_benchmark.py`: states its device.
- One test is replaced: it asserts the preference being removed.
- **Not touched:** the bare tensor factory calls. They already follow the default
  and are the only part of the library obeying the rule this change states.
  Pinning them would break it. An entry point that gains a device argument
  forwards it to them rather than resolving it first.
- **Not touched:** the notebooks. Five sites there write
  `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` — the
  duplication `accelerator()` exists to remove, and each would quietly move to
  the host. They are documents rather than library surface, they are not covered
  by the suite, and folding them in would mix a rewrite of narrative code into a
  behavioural change. Recorded here so the omission is deliberate.

## Relationship to `align-device-resolution`

This supersedes that change, which is withdrawn rather than reordered.

Its `MultiRegression` fix is absorbed here. Its device-versus-precision
asymmetry decision survives and is carried into design.md. Its measured noise
floor and seed spread are carried into design.md as the context for a later
regeneration.

Its central decision does not survive: it proposed one rule everywhere, resolved
through `resolve_device`, and recorded that `resolve_device` "is correct". Both
are drift. One rule everywhere must pick a single device for every component, and
each pick is wrong somewhere — accelerate everything and the 944.5 MB of VRAM
buys nothing, host everything and the 21.6× is lost. The resolver it trusted is
the component overriding the user.

Regenerating the benchmark table is left to `benchmark-reproducibility`, which
owns that work and regenerates it regardless.
