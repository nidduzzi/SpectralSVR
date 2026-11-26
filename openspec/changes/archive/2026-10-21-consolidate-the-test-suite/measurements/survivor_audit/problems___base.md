## SpectralSVR/problems/__base.py (5 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 27 | `==` to `<=` (1) | equivalent | `order` is already restricted to {1, 2} by the guard on line 24, so `<= 1` and `== 1` agree on every reachable value |
| 30 | `edge_order=2` to `1` (1) | gap | edge order only changes the two end samples of the non-periodic stencil; the only axis_diff test compares `[2:-2]` and Burgers tests trim the time boundary, so no test reads an end sample |
| 33 | `edge_order=2` to `1` (1) | gap | as line 30, for the second derivative (end samples 0, 1, -2, -1 depend on it); `[2:-2]` excludes all of them |
| 35 | remove `@abc.abstractmethod` on `generate` (1) | gap | `residual` and `spectral_residual` stay abstract so `Problem()` still raises; no test instantiates a subclass that omits only `generate` |
| 42 | `*,` to `/,` in the abstract signature (1) | equivalent | the abstract body is `pass` and never called (no `super().generate`), so the signature has no runtime effect |

Gaps
- Line 30/33: `axis_diff` non-periodic on `u = x**2` (order 1 and 2) compared over the full axis including end samples against `2x` and `2` (edge_order=2 is exact for quadratics, edge_order=1 is not).
- Line 35: a `Problem` subclass defining only `spectral_residual` and `residual` raises `TypeError` on instantiation.
