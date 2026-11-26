# Survivors over `SpectralSVR/basis/_nufft.py`

Second run at `4cdb0bd`, 117 mutants, each against the tests that cover its line (the archived consolidation's `selected_pytest.sh`); 5 survive. The first run at `c452976` (115 mutants) also left the grid factor `2 * m` turned into `3 * m` alive; the spy test of `4cdb0bd` kills it.

| line | mutant | verdict | reason |
| --- | --- | --- | --- |
| 27 | `except ImportError` replaced | dead | torchkbnufft is a hard dependency, so the import cannot fail |
| 28 | `return False` to `True` | dead | same branch |
| 62 | `domain[d][0]` to `domain[d][1]` | equivalent | shifts the trajectory by exactly one period, `2 pi`, which the transform wraps |
| 62 | `domain[d][0]` to `domain[d][-1]` | equivalent | `[-1]` of a pair is `[1]`, the same one-period shift |
| 65 | `not` inserted in the trajectory dtype choice | dead | only a complex query point reaches it, and evaluation points are real |
