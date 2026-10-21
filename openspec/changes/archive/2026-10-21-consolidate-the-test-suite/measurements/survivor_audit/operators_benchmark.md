## SpectralSVR/operators/benchmark.py (2 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 53 | `perf_counter() - start`->`% start` (1) | gap | `t % start == t - start` whenever the fit lasts less than the clock reading at its start (always, for a monotonic clock counting from boot), and the sleep-based test only bounds the value; patch `time.perf_counter` to return 1.0 then 4.0 and assert `fit_seconds == 3.0` |
| 84 | `groupby(..., sort=False)`->`sort=True` (1) | gap | `summarise` row order is never asserted (the record test indexes by key); pass runs with datasets/operators in non-alphabetical order and assert the output order follows first appearance |

### Gaps

- 53: scripted clock (`perf_counter` returns 1.0 then 4.0) gives `fit_seconds == 3.0`.
- 84: `summarise` on runs ordered ("z", then "a") returns index order z, a.
