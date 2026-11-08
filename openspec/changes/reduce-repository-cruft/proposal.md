## Why

The repository carries working debris that published research code usually does not: mutation results kept as hand-written verdict logs keyed to line numbers, four near-duplicate cosmic-ray configs, six untracked mutation databases at the root, 25 notebooks with committed outputs that nothing executes, and no CI. The gate runs only on one machine, and nothing identifies the commit behind the paper's numbers. Each item costs little alone. Together they make the claims harder to check, not easier.

## What Changes

- **CI.** A GitHub Actions workflow runs `just check` on every push and pull request, on CPU, with a job timeout and a fixed worker and thread count.
- **Citation and provenance.** A `CITATION.cff` at the root. An annotated tag scheme marks the commit whose benchmark record the paper reports. The local `backup/pre-replay*` tags are removed once confirmed absent from the remote.
- **One mutation-testing setup.** One cosmic-ray config per mutation target under `mutation/`, with no resource wrapper copied into each, behind `just mutate <target>`. Mutation databases and reports are written outside the tree. The four root configs and the `mutate-changed` recipe's root databases go.
- **Equivalent mutants as data.** Mutants judged equivalent are listed in one tracked registry, each with a one-line reason, and skipped by a filter. A registry entry that no longer matches the source fails the suite. This replaces `survivors.md`-style prose logs for future changes. A manually triggered workflow runs a mutation target and keeps the report as a CI artifact, not a committed file.
- **Notebooks.** A small curated set moves to `examples/`, has its outputs stripped, and is executed by the test suite on CPU. Scratch notebooks are deleted: `notebooks/test.ipynb`, `notebooks/solver1.ipynb`, `notebooks/memory_test.ipynb`, `notebooks/cases/burgers_new/`, `notebooks/cases/weatherbench2/test.ipynb`, and `notebooks/spectralsvr_2d ls_coeff.ipynb`. Every other notebook is either kept as an example or deleted, notebook by notebook, with the user's sign-off.
- **Results location.** AGENTS.md says recorded numbers live in `results/`. They live in `benchmarks/`, and `results/` is gitignored scratch. AGENTS.md is corrected.
- **Evidence convention for later changes.** AGENTS.md states it: a measurement that sets a tolerance is a maintained script under `scripts/measure/`, and the test module that holds the tolerance imports it and runs it at a small size, so the link is code and the script cannot rot. Frozen one-off probes inside a change directory stop. Nothing existing is migrated.
- **Prose-budget ratchet.** Evaluated against ruff's rules. The design records the decision.

Out of scope: `openspec/changes/archive/` stays as it is, and so does `openspec/rewritten-commit-hashes.txt`.

## Capabilities

### New Capabilities
- `mutation-testing`: how equivalent mutants are declared with a reason, and that a declaration which no longer matches the source is refused.
- `runnable-examples`: every example notebook is executed by the test suite on CPU and is committed without outputs.

### Modified Capabilities

None.

## Impact

- **New files:** `.github/workflows/check.yml`, `.github/workflows/mutate.yml`, `CITATION.cff`, `mutation/` (target configs and the equivalent-mutant registry), the registry filter, `examples/`.
- **Removed files:** `cosmic-ray.toml`, `cosmic-ray-basis.toml`, `cosmic-ray-model.toml`, `cosmic-ray-scripts.toml`, the scratch notebooks, and any notebook the review drops.
- **Changed files:** `Justfile` (memory cap and workers overridable for CI, `mutate <target>`, `strip`), `.gitignore`, `pyproject.toml` (`nbclient`, `ipykernel` and `cosmic-ray` join the `test` extra, the `notebook` extra is trimmed to what the examples import), `AGENTS.md`, `README.md`.
- **CI cost:** the lock pins CUDA 12.8 torch wheels of about 3 GB. CI installs from the lock as it stands and caches the uv environment.
- **Recorded numbers:** none move. No library code changes.
