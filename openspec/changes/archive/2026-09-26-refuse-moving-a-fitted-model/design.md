## Context

See proposal.md - Why. Four measurements shape the approach, all taken before
proposing so nothing below is re-derived while implementing.

**Both kinds of model reach the state, and they fail differently.** After
`m.device = torch.device("cuda")` on a fitted model:

| model | fitted state left behind | what `predict` raises |
| --- | --- | --- |
| `LSSVR` | `sv_x`, `sv_y`, `alpha`, `b`, `y_indicies` | `... mat2 is on cpu, ...` |
| `FNN` | the `nn.Module` parameters | `... mat1 is on cuda:0, differ` |

Neither message mentions the setter, and neither fires at the assignment. So
the diagnostic a caller gets today depends on which model they chose, which is
the opposite of a contract.

**Nothing in the repository is refused by this change.** When this was written,
exactly one `.device = ` assignment existed across `SpectralSVR/`, `tests/`, `scripts/` and
`notebooks/`: `tests/model/test_lssvr.py:218`, on a model that has never been
fitted. `grep -rn "\.device *=" notebooks/` returns nothing. That test is since
removed, because the new module covers it for every kind.

**The base can already ask.** `MultiRegression` declares `trained` as an
abstract property, and both subclasses implement it — `LSSVR.py:165` returns
whether `alpha`, `b` and `sv_x` are set, `FNN.py:60` whether `params` is set.
So the rule needs no new abstract method and no subclass participation.

**Reads are unaffected.** `model/SpectralSVR.py:305` and `:309` read
`self.regressor.device` to place values and to build a random source. This
change does not alter what those reads return; it makes the value they read
true.

## Goals / Non-Goals

**Goals:**

- A model never reports a device its values are not on.
- The disagreement is refused where it is created, not where it is later felt.
- One rule in the base, identical across every kind of model.

**Non-Goals:**

- Moving fitted values between devices. Rejected below.
- Removing the setter. It stays; see below.
- Validating that a device index exists. `follow-the-stated-device`'s design.md
  records that as a deliberate run-time matter and that decision stands.
- Any change to construction.

*Corrected after review:* this list used to exclude "any change to what a
device read returns". That was false once the setter began resolving its
argument: after `m.device = "cpu"` a read now returns `device(type='cpu')`,
where it used to return the string. It was also false once `LSSVR` and `FNN`
were amended into scope, below.

## Decisions

### Refuse, rather than move the values

The tempting fix is for the setter to carry the fitted tensors along with
`.to(device)`. It is rejected for two reasons, and the second is the one that
decides it.

First, it is a side effect. The caller wrote one assignment; the model
reallocates every fitted tensor on another device, which for a fitted `LSSVR`
is the whole training set held as support vectors. Nothing in the assignment
says that is about to happen.

Second, it cannot be written once. The base class does not know what its
subclasses hold — `LSSVR` keeps five tensors, `FNN` keeps an `nn.Module` — so
moving them means an abstract method each subclass implements by enumerating
its own state. A subclass that adds a tensor later and forgets to list it
reintroduces exactly this defect, silently, and the base class has no way to
notice. A refusal is one rule in the base that no subclass can weaken or
forget, and it is expressed in terms of `trained`, which both subclasses
already answer.

*Alternative rejected:* move the values. It trades a loud, immediate refusal
for a quiet mechanism whose correctness has to be re-established every time a
subclass gains a field.

### Keep the setter

Deleting it would make the invalid state unrepresentable rather than refused,
which is ordinarily what this repository prefers. It is rejected here because
the property is public API and deletion offers no migration: an existing
`m.device = ...` becomes `AttributeError`, which says nothing about what to do
instead. The refusal says exactly what to do instead, and on an untrained model
the assignment keeps working, which is the only use of it that exists.

The guarantee sought here is also not one a type can carry. Whether a model is
trained is runtime state that changes during its life, so no annotation can
separate a model that may be moved from one that may not.

### The message has to carry the remedy

A caller who reaches this refusal has already fitted or loaded a model on a
device they did not want. Telling them the move is refused leaves them where they were. The
message therefore names three things: the device the model was fitted on, the
device it was asked to move to, and where the device is stated instead. That
third part is the only one that gets them out, so it names both places a
model comes from: `Kind(device=...)` and `Kind.load(..., device=...)`. The
model cannot tell which one produced it, and a remedy naming only the
constructor sends a caller holding a loaded model somewhere they cannot go.

This follows the shape already set by `refuse_a_generator_placed_elsewhere`,
which names both devices and which of the two moves.

### Trained, not "holds tensors"

The condition is `self.trained`, the property the base already declares, rather
than an inspection of whatever attributes happen to be set. Inspecting
attributes would make the base depend on subclass internals, which is the
coupling the previous decision exists to avoid. It also gives the two rules one
vocabulary: a model is trained, and a trained model does not move.

### A move to where the values already are is accepted

Refusing every assignment once trained made `m.device = "cpu"` on a model
fitted on `cpu` fail with "fitted on cpu and cannot move to cpu". The
requirement refuses a change that cannot be honoured, and one that moves
nothing can be. So the setter refuses only when the requested device differs
from the fitted one.

The comparison is between the memory each device names, not the text. An
unstated index is resolved to the one in use before comparing, so `cuda` and
`cuda:0` agree exactly when `cuda:0` is in use. This is deliberately stricter
than `refuse_a_generator_placed_elsewhere`, which lets an unstated index agree
with any stated one. Reusing that rule here would accept `cuda:1` for a model
whose values sit on `cuda:0`, and the model would then report a device its
values are not on, which is the untruth this change exists to remove. The
generator check is left as it is. Whether its leniency is right is a question
for that check, not a change to make quietly from here.

### Fitted state is assigned together, after the solve

The review found `trained` answering false for a model that held values:
`LSSVR._optimize_parameters_and_set` assigned `sv_x`, `sv_y` and `y_indicies`
before the solve and `alpha` and `b` after it. A fit that failed between them
left `sv_x` set and `trained` false, so the refusal accepted a move and the
model reported `cuda` with `sv_x` on `cpu`. The same ordering let a failed
re-fit pair a new `sv_x` with the old `alpha` while `trained` stayed true.

Now everything is computed into locals and assigned in one place once the solve
has returned. A fit that fails leaves the model exactly as it was.

One reader stood in the way. The `K` property infers `sigma` for the `rbf` and
`poly` kernels from `self.sv_x` on first use, and the fit's first use is
building its own Gram matrix. Deferring `sv_x` alone would therefore have
switched every unstated `sigma` to the `1.0` fallback. That includes the
benchmark's `LSSVR(kernel="rbf", C=50.0)`, and it would have happened silently.
So the fit infers `sigma` from the `X` it was given, using the same expression,
into a local copy of `kernel_params`. From that copy it builds its own kernel and
passes it to `_batched_K`, which now takes the kernel as an argument rather than
reading `K`. The copy is committed with the rest of the fitted state after the
solve. Inferring it straight into `kernel_params` would have let a failed fit
leave behind a `sigma` inferred from the inputs that failed. The test caught
exactly that against the old code, for an `rbf` model. The infer-once behaviour
is kept: a `sigma` already inferred or stated is not recomputed on a later fit.

Because this sits on the path of every reported SpectralSVR row, its safety is
shown, not argued. Predictions are compared bit-for-bit against a base built
from a detached worktree with its own environment.

`FNN` needs the same change, contrary to what this design first said. It had
excluded `FNN` on the grounds that its failed fit leaves `params` on
`self.device`, so its device never lies. Implementing the scenario measured
otherwise: a fresh `FNN` whose fit failed reported `trained` true with an
untrained network, and a failed re-fit replaced the network it had, so its
predictions changed. The scenario covers every kind of model, so `FNN` now
builds and trains its network in locals and assigns it once training has
returned. `_build_network` returns the network rather than storing it, and a
single method sets every fitted attribute together. Both fit and `load` use it.

`LSSVR.K` is left as it is, and two things about it are recorded here rather
than fixed. First, reading `K` before any fit writes `sigma = 1.0` into
`kernel_params`, so a later fit silently uses 1.0 instead of inferring from
data; `test_lssvr_kernel_property_before_fit_defaults_sigma` pins that write.
Second, its branch that infers `sigma` from `sv_x` can no longer be reached from
a fit, which now builds its own kernel. Making `K` a pure function of
`kernel_params` would change pinned behaviour, so it belongs to its own change
with its own proof.

### `load` states the device

The review found that neither `load` could be directed. Both built the model on
the process default, so for a loaded model the refusal's remedy could not be
taken. Both now accept `device: DeviceInput = None` and pass it to the
constructor. `FNN.load` also hands it to `torch.load` as `map_location`.
Without that, a state dict saved on an accelerator is restored to the
accelerator before it can be copied anywhere, and a host-only machine cannot
load it at all.

## Risks / Trade-offs

- **A caller who legitimately wants a fitted model elsewhere has no route.** →
  There is no such route today either; the assignment that appears to offer one
  produces a broken model. The honest answer is to state the device when
  building the model, which the message says. If a real need for relocation
  appears, it is a separate change with its own decision about side effects,
  and it will land against a setter that currently refuses rather than one that
  lies.

- **The refusal fires on code that "worked" before.** → Measured: no such code
  exists in the repository, and any that existed elsewhere was already producing
  a model whose next `predict` raises. Turning a later, misleading failure into
  an immediate, accurate one is the point.

- **`trained` is the subclasses' answer, so a subclass that answers it wrongly
  escapes the rule.** → It is already the answer the rest of the base relies on.
  This risk was first dismissed with "a subclass that reports `trained` wrongly
  is broken in more places than this". Review then found `LSSVR` doing exactly
  that after a partial fit, so the dismissal was wrong. The fix is to make
  `trained` true whenever fitted state exists. Widening the rule to inspect
  attributes would have been the wrong fix; see *Fitted state is assigned
  together*.

- **Deferring `sv_x` could move the reported numbers.** → It would have,
  through `sigma` inference. Handled above, and shown bit-identical against a
  base worktree rather than argued.

## Migration Plan

1. The setter refuses when trained; the message names both devices and the
   constructor.
2. Tests pin the message's content, the untrained path, and both subclasses.
3. Amended after review: a move to where the values already are is accepted;
   `LSSVR` and `FNN` assign their fitted state once fitting returns, with
   `LSSVR`'s `sigma` inferred from the fit's own data; both `load`s accept
   `device`, and the refusal names `load` as a remedy alongside the constructor.

Rollback is a revert. Nothing depends on the refusal existing.

## Open Questions

None. Whether to move the values, whether to delete the setter, and whether to
validate a device index were the three open questions, and each is settled
above or recorded as out of scope. Review and implementation raised five
more, each now a recorded decision. The review asked whether a no-op move is
refused (it is accepted), what a partial `LSSVR` fit may leave behind
(nothing), and whether `load` can be directed (it can). Implementation then
asked whether `FNN`'s partial state after a failed fit is in scope (it is, and
is fixed) and whether `K`'s write on read is fixed here (it is not, and is
recorded above).
