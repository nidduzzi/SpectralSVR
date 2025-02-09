# skripsi_program

## Installation
```bash
pip install -q git+https://github.com/nidduzzi/SpectralSVR.git
```
Then import as
```python
from SpectralSVR import (
    SpectralSVR,
    FourierBasis,
    StandardScaler,
    Antiderivative,
    LSSVR,
    to_real_coeff,
    to_complex_coeff,
    get_metrics,
)
```

Examples are available in the notebooks/cases/ directory. Please run block by block since some parts at the end of each notebook may be unfinished.

## Testing

If using poetry use the following command:

```bash
poetry run pytest
```

Otherwise, use

```bash
pytest
```
