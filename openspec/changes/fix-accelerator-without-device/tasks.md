## 1. Failing first

- [ ] 1.1 Add a test naming the scenario "A build that supports an accelerator the machine lacks names the host". In a subprocess whose environment sets `CUDA_VISIBLE_DEVICES=""`, set a process default device, then call `accelerator()` and `placed()` on its result, and compare both against `torch.get_default_device()` read in that subprocess. Fuzz the default over `cpu` and `meta`, the devices a machine without an accelerator has. Verify it fails on the parent commit, reproducing CI run 42507523693 (`cuda` against `cpu`).

## 2. Repair

- [ ] 2.1 Pass `check_available=True` to both `torch.accelerator.current_accelerator()` calls in `SpectralSVR/utils/__init__.py`. Verify the new test passes. Verify the full suite passes both as is and with `CUDA_VISIBLE_DEVICES=""` set inside the resource wrapper, where it gave 1 failed, 461 passed, 15 skipped.
- [ ] 2.2 Run `just check` through the resource wrapper, and record the test count and wall time. Push on the user's say-so, and record the CI result and the warm-cache wall time that task 1.2 of `reduce-repository-cruft` asks for.

## Workflow follow-up

- Open a PR stacked on `feat/reduce-repository-cruft`, and archive after review.
