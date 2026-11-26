## SpectralSVR/model/LSSVR.py (107 survivors)

Tests reached: the independent KKT test pins only `linear` and `rbf` (sigma stated, never `sigma=None`); `poly`, `tri`, `frob`, `max` appear only in smoke tests (finite output, right shape).

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 31 | poly `(gamma22*linear+1)**d`: every `+`, `*`, `**` swap, `1`->`2` (19) | gap | poly numerics never compared with anything; KKT test excludes poly |
| 39 | tri `1 - cdist/sigma`: `-`, `/` swaps, `1`->`2` (10) | gap | tri only smoke-tested |
| 43 | frob `x_i.unsqueeze(1) - x_j`: `-` to `+ * / // %` (5) | gap | frob only smoke-tested |
| 47 | max `x_i.unsqueeze(1) - x_j`: `-` to `+ * / // %` (5) | gap | max only smoke-tested |
| 56 | `2**21 // dims + 7`: every operator swap, `2`->`3` (24) | gap | the kernel matrix is value-invariant under the block size (batched test uses `lambda dims: 1`), so only the sizing contract in the docstring tells them apart; no test calls `default_batch_size` (`/`-forms give a float step so `range` raises once n exceeds it; `-7`, `>>`, `//7` give 0 or negative for large dims) |
| 72 | poly `d` default `3.0`->`4.0` (1) | gap | docstring documents `d` default 3; every test passes `d=2.0` |
| 73 | poly `gamma22=(2*sigma)**-2`: `*`, `**`, unary `-` swaps, `2`->`3` (15) | gap | poly numerics unpinned (as line 31); the `sigma` default here is also unreachable from LSSVR (K sets 1.0 first) |
| 77 | rbf `sigma` default `1.0`->`2.0` (1) | dead | LSSVR always supplies sigma (K sets 1.0 before fit, `_optimize_parameters_and_set` computes it); only a direct `torch_get_kernel("rbf")` call without sigma reaches it |
| 80 | tri `sigma` default `1.0`->`2.0` (1) | gap | `LSSVR(kernel="tri")` without sigma reaches it (tri has no data-derived sigma); tests always pass sigma. Default undocumented, low priority |
| 139 | `C=1.0`->`2.0` (1) | gap | docstring documents default C=1.0; default-C fits are only checked for shape/finite (`test_lssvr_predict_numpy_roundtrips`, `fit_accepts_numpy`) |
| 168 | `trained`: `and`->`or` on `alpha`/`b` (1) | dead | alpha, b, sv_x are set together by fit and load; a partial state needs manual attribute assignment |
| 169 | `trained`: `and`->`or` on `b`/`sv_x` (1) | dead | same |
| 183 | `sv_x.var(0)`->`var(1)` (1) | dead | branch needs sv_x set while sigma is None; fit stores sigma in `kernel_params`, load restores it, so only manual assignment of `sv_x` reaches it |
| 184 | `_initialized_sigma = True`->`False` (1) | equivalent | the flag only skips recomputation; next `K` call finds sigma not None and takes the `else` that sets it True |
| 189 | `_initialized_kernel = True`->`False` (1) | equivalent | rebuilds an identical kernel from unchanged `kernel_params` on each `K` access; cost only |
| 199 | `x_i.shape[1]`->`shape[0]` (1) | gap | `batch_size_func` argument unobserved (tests use `lambda dims: 1`, constant); spy `batch_size_func` and assert it is called with the feature count. Value-invariant |
| 200 | `x_j.shape[1]`->`shape[0]` (1) | gap | same, for the second argument |
| 205 | `<=` to `== != < > >= is is not`, `not`, `and`->`or` (9) | gap | both branches give the same matrix, so only which path ran differs; spy the kernel and assert one call when both sizes fit the block and ceil(ni/bi)*ceil(nj/bj) calls otherwise (include n == batch size). Value-invariant |
| 211 | `range(0,..)`->`range(-1,..)` (1) | gap | with block size > 1 rows 0..bs-2 are never written (first block slice `[-1:bs-1]` is empty) and stay uninitialised; survives only because the test uses block size 1. Use `batch_size_func=lambda d: 3`, n not a multiple of 3 |
| 213 | inner `range(0,..)`->`range(-1,..)` (1) | gap | same, along the columns |
| 227 | debug `A[1:, 1:]`->`A[2:, 1:]` (1) | equivalent | argument of `logger.debug` only; slice is valid, no state changes |
| 230 | `A[1:, 1:].shape[0]`->`A[1:, 2:].shape[0]` (1) | equivalent | column slice does not change `shape[0]` |
| 236 | `A[0, 1:] = 1`->`2` (1) | equivalent | row 0 of the bordered system is homogeneous (`B[0]=0`), so scaling it by 2 leaves the solution set, and the minimum-norm lstsq solution, unchanged (conditioning differs only in rounding) |
| 262 | `arange(0, y.shape[1])`->`arange(1, ...)` (1) | gap | `y_indicies` is a documented attribute never read; assert it equals `arange(n_outputs)` after fit and after a dump/load round trip |
| 270 | `_predict` assert `and`->`or` (1) | dead | unfitted model has all three None so the assert still fires; partial state needs manual assignment |
| 291 | `get_p_matrix` assert `and`->`or` (1) | dead | same; unfitted has both None |
| 338 | `load` type check `!=`->`<` (1) | gap | the only mismatching payload tested has type "FNN", which sorts before "LSSVR"; load a payload whose type sorts after it (e.g. "SVR", "ZZZ") |

### Gaps

- 31, 73: fit `poly` with an independent KKT solve (extend `gram_matrix` with `(x z^T / (2 sigma)^2 + 1)**d`) for several `d` and sigma.
- 39, 43, 47: add `tri`, `frob`, `max` to the KKT test with closed-form Gram matrices (`1 - cdist/sigma`, `||x_i - x_j||_2`, `max_k(x_ik - x_jk)`).
- 56: property test on `default_batch_size(dims)` for `dims` in 1..2**22: positive int and `2**21 <= bs*dims <= 2**21 + 7*dims`.
- 72: `LSSVR(kernel="poly")` without `d` equals the KKT/poly oracle with `d=3`.
- 80: `LSSVR(kernel="tri")` without sigma equals the oracle with sigma 1 (low priority).
- 139: `LSSVR(kernel="linear")` without `C` equals the KKT solve with C=1.
- 199, 200: spy `batch_size_func` and assert it is called with `x.shape[1]` for both operands.
- 205: spy the kernel passed to `_batched_K` and assert the call count at and either side of the block size.
- 211, 213: batched vs unbatched `_batched_K` with `batch_size_func=lambda d: 3` and n not divisible by 3.
- 262: `y_indicies == arange(o)` after fit and after load.
- 338: `LSSVR.load` of a payload typed "ZZZ" raises the mismatch error.
