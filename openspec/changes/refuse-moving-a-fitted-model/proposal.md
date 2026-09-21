## Why

A fitted model will report a device its tensors are not on, and say so without
complaint:

```
m = LSSVR(kernel="linear"); m.fit(X, y)   # sv_x on cpu
m.device = torch.device("cuda")           # accepted
m.device      -> cuda
m.sv_x.device -> cpu
```

The assignment is the mistake, and it passes silently. What fails is `predict`,
later, with `RuntimeError: Expected all tensors to be on the same device, but
got mat2 is on cpu, ...`. The message names neither the setter nor the
assignment that caused it, and it reaches whoever calls `predict` rather than
whoever wrote the line.

Both subclasses reach the state, and they fail differently, so a caller cannot
even learn one signature for it:

| model | after the assignment | at `predict` |
| --- | --- | --- |
| `LSSVR` | `device` says `cuda`, `sv_x` on `cpu` | `... mat2 is on cpu, ...` |
| `FNN` | `device` says `cuda`, params on `cpu` | `... mat1 is on cuda:0, differ` |

`follow-the-stated-device` made every device answer in the library the
caller's. This is the one place left where the library gives a device answer
that is not true.

## What Changes

- The `device` setter on `MultiRegression` refuses when the model is trained,
  naming the device it was fitted on, the device it was asked to move to, and
  the constructor as the place to say it instead. An untrained model is
  unaffected.
- The setter is **kept**. Deleting it would remove public API with no migration
  path, and the guarantee wanted here is a refusal, not an absence.
- The fitted tensors are **not** moved. Moving them is a silent side effect on
  a caller's model, and it would have to be written per subclass, where a
  subclass that forgets one tensor reintroduces exactly this bug quietly. A
  refusal is one rule in the base class that no subclass can weaken.

Measured, so the cost is known rather than hoped for: exactly one `.device = `
assignment exists in the whole repository, `tests/model/test_lssvr.py:218`, and
it is on an **untrained** model. No notebook assigns `.device`. Nothing that
exists today is refused by this change.

## Capabilities

### New Capabilities
<!-- None. The rule belongs to the capability that already governs who decides
     where work is placed. -->

### Modified Capabilities
- `device-selection`: gains a requirement that a model never reports a device
  its fitted state is not on, and that a device change it cannot honour is
  refused at the assignment rather than noticed later. The capability is
  introduced by `follow-the-stated-device`, this change's parent, and is
  extended rather than altered — no requirement it states is changed.

## Impact

- `SpectralSVR/model/__base.py`: the `device` setter on `MultiRegression`.
- **Not touched:** `LSSVR` and `FNN`. Both already implement the `trained`
  property the base declares, so the base can ask whether a move is honourable
  without a new abstract method and without either subclass enumerating its own
  fitted tensors.
- **Not touched:** every read of `.device`. `model/SpectralSVR.py:305` and
  `:309` read `self.regressor.device` and are correct; this change makes that
  value trustworthy rather than changing what it returns.
- **Not touched:** a device index that does not exist. `LSSVR(device="cuda:99")`
  constructs and raises `AcceleratorError: CUDA error: invalid device ordinal`
  only at `fit`. `follow-the-stated-device`'s design.md records that as a
  deliberate run-time matter, because device count is a run-time fact and
  torch's own message is already specific. That decision stands; it is named
  here so the omission is deliberate rather than overlooked.
- One test gains coverage; none is expected to change.
