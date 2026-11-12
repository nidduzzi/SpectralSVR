## 1. Failing first

- [x] 1.1 Add a test naming the scenario "A build that supports an accelerator the machine lacks names the host". In a subprocess whose environment sets `CUDA_VISIBLE_DEVICES=""`, set a process default device, then call `accelerator()` and `placed()` on its result, and compare both against `torch.get_default_device()` read in that subprocess. Fuzz the default over `cpu` and `meta`, the devices a machine without an accelerator has. Verify it fails on the parent commit, reproducing CI run 42507523693 (`cuda` against `cpu`). Done: `test_a_machine_without_the_accelerator_names_its_default_device`, marked `oracle_independent` because the expected device is torch's own report in the subprocess. Before the repair the subprocess exits with `RuntimeError: No CUDA GPUs are available`: `accelerator()` names `cuda`, and `placed()` then asks for the index of a device that does not exist.

## 2. Repair

- [x] 2.1 Pass `check_available=True` to both `torch.accelerator.current_accelerator()` calls in `SpectralSVR/utils/__init__.py`. Verify the new test passes. Verify the full suite passes both as is and with `CUDA_VISIBLE_DEVICES=""` set inside the resource wrapper, where it gave 1 failed, 461 passed, 15 skipped. Done: the new test passes in 5.9 s. With `CUDA_VISIBLE_DEVICES=""`: 452 passed, 15 skipped. That is fewer than CI's 477 because this branch sits below `reduce-repository-cruft` and lacks its tests.
- [x] 2.2 Run `just check` through the resource wrapper, and record the test count and wall time. Gate done: ruff, mypy and ty clean, 467 passed in 48.87 s, 51 s wall. CI on a CPU-only runner, through `feat/reduce-repository-cruft` which carries this fix: run 42556766375 (cold cache) 463 passed, 15 skipped, install 35 s, gate 2 min 9 s, 2 min 54 s total; run 42802039682 (warm cache, hit) 463 passed, 15 skipped, install 19 s, gate 3 min 7 s, 3 min 28 s total. The warm run's pytest took 160 s against 108 s cold, which is runner variance rather than the cache. The fifteen skips are the GPU-only tests; `TestDeviceStatements`, which failed in run 42507523693 before this fix, passes.

## Workflow follow-up

- Archived before review at the user's choice, then a PR stacked on `feat/fix-real-derivative-at-nyquist`, with `feat/reduce-repository-cruft` rebased onto this branch.
