# Survivor audit: small modules

Population: the mutants that survived the final phase-1 to phase-4 runs, one per module, operator and source line (2776). Each was rerun against the whole suite at `30b88d4` (`cosmic-ray/prepare_survivor_audit.py`, sessions `phase5audit-*`, Hypothesis seed 0). The rows below are the ones that survived that run, listed by `cosmic-ray/list_audit_survivors.py`, and classified by reading the source and the tests. A verdict is my reading, not a proof: "equivalent" says no input can distinguish the mutant, "gap" says an input exists that no test uses.

Verdicts: **equivalent**, **dead** (unreachable or unsupported input), **gap** (a test is missing), **deferred** (a known defect the consolidation does not pin).

## basis/config.py (16 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 32 | `frozen=True` to `False` | gap | no test assigns to a `SpectralConfig` field and expects `FrozenInstanceError`; AGENTS.md states the config is a frozen value |
| 51 | `@property` removed from `real_dtype` | gap | nothing in the library or the tests reads `real_dtype`; AGENTS.md documents it as the derived real precision |
| 59 | `empty(0)` to `empty(1)` | equivalent | only the dtype of the empty tensor is read |
| 106 | `object \| None` operator swaps (11) | equivalent | annotation under `from __future__ import annotations`, never evaluated |
| 108 | `is not None` to `is None`, and `not ... is not None` | gap | `ResolvedCoeffDtype` read through an instance is only reached for an empty basis (`__base.py:158`); no test builds an empty basis at a non-default precision |

## basis/sampling.py (15 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 34 | `@property` removed | equivalent | `Protocol` member declaration, checked structurally |
| 80, 81 | `ClosedUniform.is_periodic` and `supports_fft` flipped | deferred | the flags of `ClosedUniform` are the inconsistency already on the deferred list; pinning them would pin the quirk |
| 94 | `\|` operator swaps (11) | equivalent | annotation, lazily evaluated |
| 108 | `!=` to `is not` on `len(schemes)` and `ndim` | dead | identical for axis counts under 257 |

## basis/_nufft.py (4 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 25 | `except ImportError` replaced | dead | `torchkbnufft` is a hard dependency; the branch is marked no cover |
| 26 | `return False` to `True` | dead | same branch |
| 60 | `domain[d][0]` to `domain[d][1]` in the trajectory | equivalent | shifts the trajectory by exactly one period, `2 pi`, which the transform wraps |
| 63 | `not` inserted in the dtype choice | dead | only a complex query point reaches it, and coordinates are real |

## basis/domain.py (14 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 20 | `\|` operator swaps (11) | equivalent | annotation, lazily evaluated |
| 29 | `== 0` to `<= 0` | equivalent | a length is never negative |
| 29 | `== 0` to `< 0` and to `== -1` | gap | the empty domain is documented to yield no axes (`domainInputType_to_tuple((), ())` returns `()`), and no test passes one; the mutants send it to `domain[0]` |

## model/FNN.py (8 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 31 to 36 | default `MAX_EPOCH`, `lr`, `batch_size`, `n_hidden`, `w_hidden` changed by one step (5) | gap | no test reads the constructor defaults, and the tests that train pass explicit values |
| 59 | `@override` removed | equivalent | typing only |
| 103 | `requires_grad_(False)` to `True` after fitting | gap | no test checks that a fitted network is frozen |
| 166 | `type != "FNN"` to `> "FNN"` | gap | the mismatch test's other kind, `"LSSVR"`, sorts above `"FNN"`; a name sorting below it would pass the check |

## operators/__base.py (1 survivor)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 30 | `@property` removed | equivalent | `Protocol` member declaration |

## operators/dataset.py (7 survivors)

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 41 | `len(domain) == 0` to `<= 0` | equivalent | a length is never negative |
| 54 | `dim=-1` to `+1` and to `1` (2) | equivalent | the stacked tensors are one-dimensional, so the result has two axes and `-1` is `1` |
| 83, 88 | `!=` to `is not` on shape sizes (2) | gap | identical below 257, wrong above it; a dataset of more than 256 sensors or functions is refused by the mutant, and no test builds one |
| 86, 90 | `shape[0]` to `shape[1]` inside the error message (2) | gap | the message text is not asserted |

## operators/spectralsvr.py (0 survivors)

Every mutant of the one-mutant sample is killed.
