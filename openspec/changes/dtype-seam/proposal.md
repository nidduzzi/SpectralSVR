# A configurable precision seam for spectral coefficients

## Why

Coefficient precision is currently a hardcoded class constant. `FourierBasis.coeff_dtype = torch.complex64` is global per basis class, so there is no way to run a problem in double precision — and that is what blocked the fourth-order convergence anchor in `fix-etdrk4-stage-times-and-complex-weights`, which had to be written against the integrator directly because a round-trip through the library saturates at float32's floor.

Three partial seams already exist and do not meet:

- `Basis.coeff_dtype` is declared abstract (`basis/__base.py:89`), enforced (`:102-103`) and used to allocate (`:154`) — but its only implementation is a constant.
- `MultiRegression.dtype` (`model/__base.py:43`) defaults to `torch.float32` and threads to four `.to(device, dtype)` sites; `LSSVR` repeats the default at `:140`.
- `to_real_coeff` / `to_complex_coeff` bridge complex coefficients to interleaved real pairs at ten model boundaries.

The third is already sound and needs no change: it is bit-exact and dtype-transparent in both directions at 2-D and 3-D, and `StandardScaler` carries complex128 end to end with inverse error `4.4e-16`. The first two are the gap, and they are not independent — a complex128 basis feeding a default regressor **silently truncates to float32** at `model/SpectralSVR.py:273`, which passes only the device and lets the regressor's own cast do the rest.

Torch support was probed and is complete for everything the library uses, including the one genuine third-party risk:

| operation | complex128 |
| --- | --- |
| `fft` / `ifft` / `fftn` | works |
| `torchkbnufft` KbNufft forward | works |
| CUDA `fft`, CUDA `matmul` | works |
| autograd through complex | works |
| `linalg.solve` complex128 and float64 | works |

Two limits, neither blocking: `cdist` rejects complex, but the LSSVR kernel path runs on real interleaved data and needs float64, which works; `torchmetrics.mse` accepts complex128 and returns a meaningless complex value, but metrics run after the real conversion.

The cost is throughput, not compatibility. Measured on an RTX 5090:

| operation | complex64 | complex128 | penalty |
| --- | --- | --- | --- |
| matmul 2048² | 2.93 ms | 49.43 ms | 16.9× |
| fft 256×4096 | 0.68 ms | 1.91 ms | 2.8× |

GeForce runs FP64 at a thirty-second of the FP32 rate, so the matmul-bound kernel and solve pay heavily while the memory-bound transform barely notices. Double precision must therefore be opt-in.

## What Changes

- Coefficient precision becomes a field on `SpectralConfig`, resolved per instance, with `complex64` as the default so nothing moves unless asked.
- The basis-side and model-side precisions become one decision rather than two, closing the silent downcast at the model boundary.
- Five hardcoded bypasses fold into the seam: the class constant (`FourierBasis.py:29`), `generate_empty` (`:342`), `generate_coeff` (`:360`), the ETDRK4 fallback (`utils/__init__.py:611`), and the benchmark's GRF sampler (`scripts/run_benchmark.py:104`). The real-side defaults at `model/__base.py:43`, `model/FNN.py:35` and `model/LSSVR.py:140`, plus `wave_number`'s `torch.float`, are threaded from the same decision.
- `to_complex_coeff` gains a `.contiguous()` — a transposed view currently raises `Tensor must have a last dimension with stride 1` — and an optional target dtype, since it otherwise infers from its input and a float32 detour silently lands back at complex64.
- **The default is unchanged**, so every seeded field, every benchmark row and every byte-identity guarantee holds.

## Capabilities

### New Capabilities
- `coefficient-precision`: how a caller selects the precision a basis stores and a model consumes, what the default is, what must move together, and what the library guarantees at each precision.

### Modified Capabilities
- `numerical-anchor-tests`: records that an anchor requiring more precision than the default may request it, rather than being relocated to a lower layer or abandoned.

## Impact

- `SpectralSVR/basis/config.py`: the new field and its resolution.
- `SpectralSVR/basis/FourierBasis.py`: the class constant and two generator literals.
- `SpectralSVR/model/__base.py`, `FNN.py`, `LSSVR.py`, `SpectralSVR.py`: the real-side default and the boundary cast.
- `SpectralSVR/utils/__init__.py`: `to_complex_coeff` hardening; the ETDRK4 fallback.
- `scripts/run_benchmark.py`: the GRF sampler default.
- No dependency change. No default behaviour change, so no benchmark row moves and seeded output stays byte-identical.
- `operators/deeponet.py` stays pinned to float32 — that is deepxde's constraint and affects only the baseline.
