## 0. Dependency gate

- [ ] 0.1 Confirm all three dependencies are merged beneath this branch: `align-spectral-residual-with-solver` (the Burgers MMS row moves, so the multi-seed table must be generated after it), `operator-harness-spec` (this extends the capability it states), and `time-dependent-forcing-in-numerical-burgers` (its forced dataset belongs in the same run); verify with `git log --oneline` before regenerating anything

## 1. The per-sample statistic

- [ ] 1.1 Add per-sample relative L2 — the L2 norm of the error over each sample divided by the L2 norm of that sample's target, then averaged — to the metric suite beside `rse` and `rrse`, under a name that states the convention; verify it is computed per row of the prediction tensor, not globally
- [ ] 1.2 Verify no existing metric value changes: run the metric suite on a fixed array before and after and assert every pre-existing column is identical
- [ ] 1.3 Verify the two relative-error columns differ on a case where they must — a test set with one sample of much larger magnitude than the rest — so the distinction is demonstrated rather than asserted

## 2. Provenance

- [ ] 2.1 Record the device **each stage ran on**, coefficient precision, torch version, deepxde version and git revision as columns on every run row; verify each is populated in a real run rather than left empty. A single device column is wrong today: `Burgers.generate:60` integrates on `cuda:0` through `resolve_device()` while `MultiRegression.__init__` at `model/__base.py:49` hardcodes `cpu`, so one run uses both. Verify a run whose stages differ reports both rather than one standing for the other
- [ ] 2.2 Record whether the working tree was clean at generation time; verify a deliberately dirty tree is reported as such, so a number from uncommitted code is visible
- [ ] 2.3 Verify the columns survive concatenating two runs made under different conditions, which is the reason they are columns rather than a file header

## 3. Seeds and output format

- [ ] 3.1 Change the benchmark output to long format, one row per dataset, operator and seed; verify a single-seed run reproduces today's values on every column that already existed **to within the harness's measured run-to-run noise**, not exactly. *Corrected from this task as first written:* six consecutive runs of identical code vary by up to `2.51e-05` relative, so exact reproduction is not achievable and asserting it would fail for reasons unrelated to the format change. Record the observed difference against that floor
- [ ] 3.2 Add a fixed seed set as the default with the seeds recorded in the output; verify a partial run can be appended rather than requiring a restart, since DeepONet trains for ten thousand iterations per dataset per seed
- [ ] 3.3 Add aggregation at read time producing a centre and spread per metric; verify the aggregate is derived from the committed long file rather than recomputed from the models

## 4. Timing

- [ ] 4.1 Record what each operator's fit covers — a direct solve or a fixed iteration count — alongside `fit_seconds`; verify both operators report it
- [ ] 4.2 State in the benchmark documentation that the timing column measures different work for different operators and does not support a like-for-like cost comparison; verify the wording matches what the harness capability already says about the comparison's limits

## 5. One command, one record

- [ ] 5.1 Add a `bench` recipe to the `Justfile` regenerating the table with the default seed set and datasets; verify `just bench --help` or the recipe listing shows it and that it needs no remembered arguments
- [ ] 5.2 Point `notebooks/comparisons/deeponet_vs_spectralsvr.ipynb` at the committed results instead of recomputing; verify it runs without fitting any operator
- [ ] 5.3 Verify `results/` is still gitignored for scratch output while the committed table lives at a path that is tracked, so the record and the scratch cannot be confused

## 6. Regeneration and verification

- [ ] 6.1 Regenerate over the full seed set with all datasets including the forced-Burgers one; verify every row carries complete provenance and no metric is NaN
- [ ] 6.2 Record in the change which figures moved relative to the previous single-seed table and why — the MMS row from the residual fix, everything else from seed averaging — so the discontinuity is documented rather than discovered later
- [ ] 6.3 Run `just check`; verify clean
- [ ] 6.4 Verify no source file gained an explanatory prose comment; the provenance columns and the metric name must carry their own meaning
