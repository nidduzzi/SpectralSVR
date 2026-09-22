## Context

See proposal.md - Why. Four measurements shape every decision below.

**A stated host is indistinguishable from an unstated one.** `get_default_device()`
returns `cpu` both when the process pinned it and when nothing was set. The only
signal is `torch._C._len_torch_function_stack()`, which reads `0` against `1` —
private, and it counts any torch function mode, so an unrelated mode registers as
a pin. There is no public way to tell the two apart.

**The cost of the seam change is one test.** With resolution following the
default: 324 passed, 1 failed, the failure being the test that asserts the
preference being removed.

**Placement is worth a great deal at scale and little at benchmark size**, in
direction and order of magnitude. Host timings are not reproducible constants:
across two runs of the same measurement on an idle machine, `LSSVR.fit` at
`(400,128)` read 1.85 ms then 1.26 ms on the host, flipping the winner, and the
ETDRK4 solve at `(500,64,800)` read 4880 ms then 2278 ms. A median can be six
times the best of the same seven runs. Accelerator timings repeat to a decimal.
So the shape is the finding and a figure is not:

| measurement | host | accelerator | direction |
| --- | --- | --- | --- |
| `LSSVR.fit (120,64)` | sub-ms | sub-ms | host, ~1.4× |
| `LSSVR.fit (400,128)` | ~1–2 ms | ~1.3 ms | within noise, no winner |
| `LSSVR.fit (2000,512)` | ~27–35 ms | ~8.5 ms | accelerator, 3–4× |
| ETDRK4 `(50,16,100)` | ~16 ms | ~29 ms | host, ~1.7× |
| ETDRK4 `(500,64,800)` | ~2.3–4.9 s | ~0.23 s | accelerator, 10–20× |
| generate `n=800, (128,128)` | — | ~1 GB peak for ~320 MB held | 3× the held size |

The crossover moves with feature width rather than sample count, and sits inside
the noise where it falls, so it is a region and not a threshold. Nothing in this
change depends on where exactly it lies — which is the point of refusing a size
heuristic.

**A generator's device is invisible to a type checker.**
`type(torch.Generator()) is type(torch.Generator(device="cuda"))` is `True`.

**An unstated index is not the same as a stated one.** `torch.device("cuda")`
carries `index=None`; `torch.get_default_device()` under a `cuda` pin returns
`cuda:0`, so the two are not `==`. The refusal therefore compares type first and
indices only where both are stated: `cuda:0` agrees with `cuda`, and `cuda:0`
disagrees with `cuda:1`. Comparing type alone admitted the second pairing, which
is a real disagreement -- separate memories, and a draw on one unreadable from
the other.

**The test harness already neutralises the condition.** `tests/conftest.py`
carries an autouse fixture that snapshots and restores the process default around
every test, and its docstring names the cause: importing deepxde sets the default
to cuda, which would otherwise leak into every later test. So the suite is
insulated from the state the benchmark dies in — which is why a green suite and a
broken script coexisted. A new test does not need its own restoration; it needs to
confirm the fixture's reach.

## Goals / Non-Goals

**Goals:**

- The device the caller or the process states is the device used.
- Placement statable at every entry point that places work.
- One name for "this machine's accelerator", replacing a hardcoded vendor check.

**Non-Goals:**

- The bare tensor factories. They already follow the default and are the only
  part of the library obeying the rule this change states.
- Per-component accelerator defaults. The first measurement makes them
  unimplementable without overriding a stated host.
- Regenerating the benchmark table. `benchmark-reproducibility` owns it.
- Changing what an explicit device argument already does.

## Decisions

### Follow the process, hold no preference

Resolution consults a stated device, then the process default, and stops. The
library adds no third answer.

The temptation is a per-component default — accelerate the solve, host the bulk
data — which is what the original intent described and what the measurements
above support on their face. It cannot be built. A process default naming the
host reads identically to no default at all, so "accelerate unless told
otherwise" would override a caller who asked for the host, which is the defect
this change exists to remove. The choice is between a preference that silently
overrides and no preference at all, and only one of those is honest.

The intent survives, relocated: an application states the placement it wants per
call, which is what the new device argument is for.

*Alternative rejected:* detect a pin through the private mode stack. It is
private, and it counts unrelated modes, so it answers a different question than
the one asked.

### Two functions, not a closed union

`resolve_device` applies precedence to what it is given; `accelerator()` names a
device a caller may give. Opposite directions across the same boundary.

A closed union of `Stated | Accelerator | ProcessDefault` was considered because
it is this repository's idiom. It does not fit. That idiom earns its place where
enumerating the alternatives is exhaustive and a member would otherwise claim a
guarantee it cannot deliver — a bare mode count cannot express a band that keeps
conjugate pairs together, so the type carries what the value cannot. A device
carries no such guarantee, and the alternatives are not closed: every valid
device is a legitimate member. The union would wrap an already well-formed value
type, cost an import at every call site, and break `device="cuda:1"`, the form
torch itself accepts. Accepting strings as well would leave it not closed, which
removes the only reason to have had it.

Measured against the second criterion it also earns nothing: of the four invalid
states available here — a malformed device name, a nonexistent device index, an
accelerator requested where none exists, and a random source on another device —
the union prevents none.

### Keep the caller's generator, refuse a disagreement at run time

A mismatch cannot be refused by a type. Both generators are the same type; the
device is a runtime attribute.

Making it type-visible means taking a seed instead of a generator and
constructing the source on the resolved device, which does make the invalid state
unrepresentable. It is rejected here for two reasons: it removes a caller's
ability to thread one generator through several calls to get a single stream, and
offering both a seed and a generator would be two ways to do one thing.

So the guarantee is a run-time refusal, and the message carries what the type
cannot: both devices, and which of the two the caller is expected to move. Torch's
own `Expected a 'cuda' device type for generator but found 'cpu'` does not say
that the generator is what moves.

What the type layer does still catch: a malformed name raises at
`torch.device("gpu")` construction. A nonexistent index constructs and fails
later, so it is a run-time matter like the generator.

### The library builds its own random source where the work is

`inverse` and `inverse_coeff` construct `torch.Generator().manual_seed(42)` when
the caller supplies none — on the host, with a fixed seed, feeding a draw that
takes no device. Under a stated accelerator that is the disagreement this change
refuses, raised from inside the library against a source the caller never chose.
A refusal is owed to a caller who made a choice; here nobody did.

So where the library builds the source, it builds it where the work is, and the
refusal is reserved for a source the caller supplied. The two rules are the same
rule seen from either side of the boundary: the source and the work agree, and
whoever created the source is responsible for making that true.

*Alternative rejected:* leave it and let the refusal fire. It would report the
library's own omission as the caller's error, and the message would name a device
the caller never mentioned.

### A device difference is reconciled, a precision difference is not

Carried from the superseded `align-device-resolution`, where it was reasoned and
remains correct.

`coefficient-precision` requires that a basis and a regressor disagreeing on
precision is refused or reported. Device does not deserve the same treatment, and
the asymmetry is the point rather than an inconsistency. Moving a tensor between
devices preserves every value; reducing its precision destroys information that
cannot be recovered. A warning is owed to a caller about to lose something.
Nothing is lost by a transfer, so refusing one would reject a working
configuration to no purpose. `model/SpectralSVR.py` already moves values with
`.to(self.regressor.device)`, which stays.

The generator refusal above is not an exception to this. A generator is not moved
because moving it changes the numbers it draws, which is a loss.

### Every entry point that places work takes a device, including the lower ones

The requirement is that placement is statable wherever work is placed, and the
basis generation entry points place work while accepting no device. Adding the
argument only to the problem generators would leave the requirement stated and
unmet: a caller reaching the basis directly could direct placement only by moving
the whole process, which is the coarse instrument this change exists to avoid.

The argument is forwarded to the factory rather than resolved on the way, per the
decision below, so a lower entry point that receives nothing behaves exactly as
it does today.

*Alternative rejected:* narrow the requirement to the entry points already in
scope. It would have made the spec describe the change rather than the behaviour,
and the gap would have survived as debt the coverage guard cannot see, because it
keys on scenarios rather than on requirements.

### Resolution is for materialising, not for forwarding

A factory given `device=None` resolves it itself: `torch.zeros(3, device=None)`
lands on the default. So a path that only forwards a device into a factory needs
no resolution and should pass `None` through.

Resolution is needed where a concrete device must exist: `.to(device)`, where
`None` is a no-op rather than a move, and anywhere a device is stored on an
object. Both current call sites are the former kind.

## Risks / Trade-offs

- **Work moves off the accelerator by default.** The solve currently runs there
  and will run on the host unless asked. → Measured: metrics identical to about
  six significant figures, `r2` differing in its last digit; and the benchmark's
  own sizes are in the region where the host is 1.7× faster. A caller with a
  larger problem can now ask, which they could not before.

- **A caller relying on the old automatic accelerator loses it silently.** →
  There is no way to warn without a preference to compare against. The replaced
  test states the new rule, and the argument makes the old behaviour available in
  one word.

- **`accelerator()` returns a device without an index** —
  `device(type='cuda')`, which is not `==` `torch.device('cuda:0')`. → No device
  equality exists in the library; every site is `.to(device=...)`, which
  tolerates it. The one equality in the suite pins the host explicitly. A caller
  comparing a stored device would see the difference, which is why it is stated
  here.

- **Determinism.** GPU reductions are less deterministic; the run-to-run floor
  was measured at `2.51e-05` relative over six consecutive runs, against a seed
  spread running 26,010× to 180,953× that floor. → Recorded for whoever
  regenerates the table; this change moves work toward the host, so it does not
  raise that floor.

## Migration Plan

1. Replace the test asserting the removed preference, so the rule is stated
   before the code changes under it.
2. Resolution follows the default; `accelerator()` appears.
3. The device argument reaches the entry points that place work.
4. The hardcoded host in the model goes.
5. The benchmark states its device, and runs in the configuration that fails
   today.

Rollback is a revert; nothing depends on this beyond the benchmark completing.

## Open Questions

None. Whether a per-component default was possible, whether a union fit, and
whether a seed should replace the generator were the three open questions, and
the decisions above settle each with the measurement or the rule that decided it.
