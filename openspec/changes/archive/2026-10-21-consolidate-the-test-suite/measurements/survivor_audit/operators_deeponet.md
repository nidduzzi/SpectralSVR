## SpectralSVR/operators/deeponet.py (23 survivors)

The file has `from __future__ import annotations`. The environment has deepxde and a CUDA device, so the fit tests do run.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 30 | `except ImportError`->`CosmicRayTestingException` (1) | gap | reached only when deepxde is absent (`pragma: no cover`); a test can set `sys.modules["deepxde"] = None` and assert the `baselines`-extra ImportError |
| 69 | `float \| None` annotation to `+ - * / // % ** >> << & ^` (11) | equivalent | lazy annotation under `from __future__ import annotations`; a dataclass never evaluates it |
| 93 | `2 * len(harmonics)`->`2 + len`, `2 ** len` (2) | equivalent | `harmonics` is the class constant `(1, 2)` so `len == 2` and 2*2 == 2+2 == 2**2 |
| 98 | `unbind(-1)`->`unbind(+1)` / `unbind(1)` (2) | dead | identical for 2-D `(Q, d)` points, the only shape the operator passes; differs for rank >= 3 |
| 98 | `strict=True`->`False` (1) | gap | width mismatch between points and `periods` is guarded at construction, so no test calls `trunk_features` with a wrong column count; assert ValueError |
| 105 | `stack(columns, dim=-1)`->`dim=+1` / `1` (2) | dead | columns are 1-D, where dim 1 == dim -1; differs for rank >= 3 points |
| 178 | `iterations=10_000`->`10001` (1) | equivalent | default undocumented; `scripts/run_benchmark.py` and every test pass `iterations` |
| 179 | `seed=0`->`1` (1) | equivalent | default undocumented; `scripts/run_benchmark.py` and the tests' `fitted` helper pass `seed` explicitly |
| 193 | `len(periods) != trunk.shape[1]`->`is not` (1) | equivalent | small ints are identical below 257; trunk coordinate counts are 1-3 |
| 206 | `device.type != "cpu"`->`> "cpu"` (1) | equivalent | every torch device type other than `cpu` ("cuda", "mps", "xpu", "meta", ...) sorts after "cpu" |

### Gaps

- 30: with `sys.modules["deepxde"] = None`, `_import_deepxde()` raises ImportError naming `uv sync --extra baselines`.
- 98: `Lu2022Burgers(periods=(None, 1.0)).trunk_features` on points of 1 and of 3 columns raises ValueError.
