## Why

The first CI run, on a CPU-only runner, failed `TestDeviceStatements`: `accelerator()` returned `cuda` on a machine with no GPU. `torch.accelerator.current_accelerator()` reports the accelerator the torch build supports, not one that is present, unless called with `check_available=True`. The lock installs CUDA 12.8 builds, so every machine without a GPU gets `cuda` named as "the accelerator here", against the device-selection requirement that naming it yields the host where none is present. Every development machine so far had a GPU, so the fallback never ran.

## What Changes

- `accelerator()` and `placed()` in `SpectralSVR/utils` ask for an accelerator that is present, not one the build supports.
- A test names the accelerator in a subprocess that sees no CUDA device, so the fallback is checked on a host that has a GPU too.

## Capabilities

### New Capabilities

None.

### Modified Capabilities
- `device-selection`: "The accelerator is named, not detected ad hoc" gains a scenario for a torch build that supports an accelerator on a machine that has none.

## Impact

- **Code:** `SpectralSVR/utils/__init__.py`, two calls.
- **Tests:** `tests/utils/test_utils.py`.
- **Recorded numbers:** none move. The benchmark record states its devices, and every recorded row ran on a machine with a GPU, where the two calls return the same device before and after.
