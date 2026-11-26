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

- The `device` setter on `MultiRegression` refuses when the model is trained
  and the requested device names different memory from the fitted one, naming the device it was fitted on, the device it was asked to move to, and
  the constructor and `load` as the places to say it instead. An untrained
  model is unaffected.
- The setter is **kept**. Deleting it would remove public API with no migration
  path, and the guarantee wanted here is a refusal, not an absence.
- The fitted tensors are **not** moved. Moving them is a silent side effect on
  a caller's model, and it would have to be written per subclass, where a
  subclass that forgets one tensor reintroduces exactly this bug quietly. A
  refusal is one rule in the base class that no subclass can weaken.
- A direction naming the device the fitted values are already on moves nothing
  and is accepted. Devices are compared as the memory they name, so an unstated
  index means the one in use.
- The setter resolves what it is given, as construction does. It used to store
  its argument as-is, so `m.device = "cpu"` read back the string `'cpu'`.
- `LSSVR` and `FNN` assign their fitted state together, once fitting has
  returned. A failed `LSSVR` fit used to leave `sv_x` set while `trained` was
  false, so the refusal let the model be moved away from values it held. A
  failed `FNN` fit used to leave `trained` true with an untrained network, and a
  failed re-fit replaced the network it had.
- `LSSVR.load` and `FNN.load` accept `device`. Without it a loaded model sat on
  the process default, and the refusal's remedy, stating the device, could not
  be taken for it.

Measured, so the cost is known rather than hoped for: when this was proposed,
exactly one `.device = ` assignment existed in the whole repository,
`tests/model/test_lssvr.py:218`, on an **untrained** model. That test is since
removed; see Impact. No notebook assigns `.device`. Nothing that
exists today is refused by this change.

## Capabilities

### New Capabilities
<!-- None. The rule belongs to the capability that already governs who decides
     where work is placed. -->

### Modified Capabilities
- `device-selection`: gains a requirement that a model never reports a device
  its fitted state is not on, and that a device change it cannot honour is
  refused at the assignment rather than noticed later. The capability was
  introduced by `follow-the-stated-device`, since archived, and is extended
  rather than altered — no requirement it states is changed. The `load`
  scenario serves its existing requirement that placement be statable
  everywhere work is placed, which `load` did not meet.

## Impact

- `SpectralSVR/model/__base.py`: the `device` setter on `MultiRegression`.
- `SpectralSVR/utils/__init__.py`: `name_the_same_memory`, the comparison the
  setter uses to accept a direction that moves nothing.
- `SpectralSVR/model/LSSVR.py`: fitted state is assigned after the solve, and
  `sigma` is inferred from the data the fit is given rather than read back from
  `sv_x`. `scripts/run_benchmark.py` builds `LSSVR(kernel="rbf", C=50.0)` with
  no `sigma`, so this sits on the path of every reported SpectralSVR row. The
  rows must be shown unchanged, not assumed unchanged.
- `SpectralSVR/model/FNN.py`: the network is built and trained in locals and
  assigned once training returns. Seeded predictions were shown bit-identical
  to the previous code, and dumps written by the previous code still load.
- `SpectralSVR/model/LSSVR.py`, `SpectralSVR/model/FNN.py`: `load` accepts
  `device`.
- The refusal itself still needs no subclass participation. Both kinds already
  implement the `trained` property the base declares.
- **Not touched:** every read of `.device`. `model/SpectralSVR.py:305` and
  `:309` read `self.regressor.device` and are correct; this change makes that
  value trustworthy rather than changing what it returns.
- **Not touched:** a device index that does not exist. `LSSVR(device="cuda:99")`
  constructs and raises `AcceleratorError: CUDA error: invalid device ordinal`
  only at `fit`. `follow-the-stated-device`'s design.md records that as a
  deliberate run-time matter, because device count is a run-time fact and
  torch's own message is already specific. That decision stands; it is named
  here so the omission is deliberate rather than overlooked.
- A new test module covers the requirement. `test_lssvr_device_setter` is
  removed, because the new module's unfitted-model test covers what it checked
  for both kinds.
