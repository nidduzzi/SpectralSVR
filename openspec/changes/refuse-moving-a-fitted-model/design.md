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

**Nothing in the repository is refused by this change.** Exactly one
`.device = ` assignment exists across `SpectralSVR/`, `tests/`, `scripts/` and
`notebooks/`: `tests/model/test_lssvr.py:218`, on a model that has never been
fitted. `grep -rn "\.device *=" notebooks/` returns nothing.

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
- Any change to what a device *read* returns, or to construction.

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

A caller who reaches this refusal has already fitted a model on a device they
did not want. Telling them the move is refused leaves them where they were. The
message therefore names three things: the device the model was fitted on, the
device it was asked to move to, and that the device is stated when the model is
built. That third part is the only one that gets them out.

This follows the shape already set by `refuse_a_generator_placed_elsewhere`,
which names both devices and which of the two moves.

### Trained, not "holds tensors"

The condition is `self.trained`, the property the base already declares, rather
than an inspection of whatever attributes happen to be set. Inspecting
attributes would make the base depend on subclass internals, which is the
coupling the previous decision exists to avoid. It also gives the two rules one
vocabulary: a model is trained, and a trained model does not move.

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
  escapes the rule.** → It is already the answer the rest of the base relies on;
  a subclass that reports `trained` wrongly is broken in more places than this.
  The rule adds no new surface, which is why the condition was chosen.

## Migration Plan

1. The setter refuses when trained; the message names both devices and the
   constructor.
2. Tests pin the message's content, the untrained path, and both subclasses.

Rollback is a revert. Nothing depends on the refusal existing.

## Open Questions

None. Whether to move the values, whether to delete the setter, and whether to
validate a device index were the three open questions, and each is settled
above or recorded as out of scope.
