## SpectralSVR/model/__base.py (40 survivors)

Tests never fit or predict with a 1-D `X`; 1-D `y` is covered. No test fits with `len(X) != len(y)`. `torch_json_encoder` is only fed builtins and a `Tensor`.

| line | mutants | verdict | reason |
| --- | --- | --- | --- |
| 21 | `==`->`>=` (1) | gap | any object from a module sorting after "torch" (e.g. `uuid.UUID`, module "uuid") takes the tensor branch and raises AttributeError (`.item()`) instead of TypeError; fuzz inputs are all "builtins" |
| 21 | `==`->`is` (1) | equivalent | `type(t).__module__` and `torch.__name__` are the same interned "torch" string (checked: `is` is True); only a hand-built equal-but-distinct string tells them apart |
| 60 | remove `@property` on abstract `trained` (1) | equivalent | abstractness is kept by `@abc.abstractmethod` and both subclasses override with their own property |
| 61 | remove `@abc.abstractmethod` on `trained` (1) | gap | nothing instantiates an incomplete subclass; assert `MultiRegression()` and a subclass lacking `trained` raise TypeError |
| 91 | `update=False`->`True` (1) | equivalent | parameter is never read, and no caller passes it |
| 104 | `x.ndim == 1`: `<`, unary `-`->`+ ~ not` / deleted, `-1`->`-2` (6) | gap | no test fits with 1-D `X`; with n >= 2 every one of these breaks or skips the reshape |
| 104 | `x.ndim == 1`->`<= 1` (1) | dead | differs only for 0-d `X`, which the original rejects at `x.shape[0]` |
| 110 | `y.ndim == 1`->`<= 1` (1) | dead | differs only for 0-d `y`, unsupported |
| 112 | `==`->`<=` (1) | gap | no test passes mismatched row counts; fit with fewer X rows than y rows must raise |
| 112 | `==`->`>=` (1) | gap | same, with more X rows than y rows (both directions needed) |
| 124 | remove `@overload` on first `predict` stub (1) | equivalent | typing-only; the later real `def predict` rebinds the name |
| 138 | torch branch `X.ndim == 1`: `<`, unary swaps, `-2` (6) | gap | no test calls `predict` with a 1-D torch tensor |
| 138 | `X.ndim == 1`->`<= 1` (1) | dead | differs only for 0-d `X`, which fails later in the original |
| 141 | numpy branch, same operators (6) | gap | no test calls `predict` with a 1-D numpy array |
| 141 | `<= 1` (1) | dead | 0-d only |
| 151 | `predictions.reshape(-1) if X.ndim == 1`: `<`, unary swaps, `-2` (6) | gap | the 1-D-in, 1-D-out contract is untested for both torch and numpy |
| 151 | `<= 1` (1) | dead | 0-d `X` never gets here |
| 154 | abstract `dump` default `only_hyperparams=False`->`True` (1) | equivalent | abstract body never runs; subclasses declare their own default |
| 157 | remove `@classmethod` on abstract `load` (1) | equivalent | subclasses override with their own `@classmethod` |
| 159 | abstract `load` default `False`->`True` (1) | equivalent | same as 154 |

### Gaps

- 21: `torch_json_encoder(uuid.uuid4())` raises TypeError.
- 61: instantiating `MultiRegression` or a subclass without `trained` raises TypeError.
- 104, 138, 141, 151: fit on a 1-D `X` (n >= 2) and predict on a 1-D torch tensor and a 1-D numpy array; the output is 1-D with matching values.
- 112: fit with `len(X) < len(y)` and with `len(X) > len(y)` raises AssertionError.
