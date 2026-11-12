# SpectralSVR

Operator learning with spectral bases: model the coefficients of a spectral
series (e.g. Fourier) with least-squares SVR / neural regressors, plus tooling
for generating PDE problem data (antiderivative, viscous Burgers).

## Installation

The project uses [uv](https://docs.astral.sh/uv/). To use it as a dependency:

```bash
uv add git+https://github.com/nidduzzi/SpectralSVR.git
# or, with pip:
pip install "git+https://github.com/nidduzzi/SpectralSVR.git"
```

For development (creates + installs the pinned environment):

```bash
just sync
```

## Usage

```python
from SpectralSVR import (
    SpectralSVR,
    FourierBasis,
    StandardScaler,
    Antiderivative,
    Burgers,
    LSSVR,
    FNN,
    to_real_coeff,
    to_complex_coeff,
    get_metrics,
)
```

Generate data with a problem, then train a `SpectralSVR` over the spectral
coefficients. `examples/` holds notebooks for the antiderivative, a manufactured
Burgers pair, and the committed comparison against DeepONet. The test suite
executes each of them on CPU, and `just strip` clears their outputs before a
commit.

Burgers data can be produced two ways:

- **Numerical simulation** via a stiff-stable ETDRK4 spectral solver
  (`Burgers().generate(..., u0=<ic>, f=0)`).
- **Method of Manufactured Solutions** (`Burgers().mms_solution(...)`), which
  returns an exact solution/forcing pair for verification.

## Testing

```bash
just check         # lint (ruff, mypy, ty), then the full suite, examples included
just test -k name  # a subset
```

Every test declares where its expected values come from with one `oracle_*`
marker, and is either fuzzed with [Hypothesis](https://hypothesis.readthedocs.io/)
or marked `no_fuzz`; `tests/conftest.py` enforces both. `AGENTS.md` states the
project's conventions in full.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).
