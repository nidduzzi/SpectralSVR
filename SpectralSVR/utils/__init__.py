import torch
import logging
import typing
from typing import Callable, TypedDict, cast, overload
from torchdiffeq import odeint
from functools import partial
from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    symmetric_mean_absolute_percentage_error,
    r2_score,
)

logger = logging.getLogger(__name__)

Number = int | float

DeviceInput = torch.device | str | None


def resolve_device(device: DeviceInput = None) -> torch.device:
    """Resolve a device, defaulting to CUDA when available else CPU.

    Centralizes device selection so it can be injected/overridden in one place
    instead of duplicating ``cuda if available`` checks across the codebase.
    """
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_complex_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_complex_coeff

    convert a tensor containing rows of real coefficient vectors into rows of complex coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p tensor where p is twice the number modes

    Returns:
        torch.Tensor -- m by p/2 tensor of elements in complex numbers
    """
    if torch.is_complex(coeff):
        logger.debug("coeff is already complex")
        return coeff
    converted_coeff = torch.view_as_complex(coeff.reshape((*coeff.shape[:-1], -1, 2)))
    return converted_coeff


def to_mag_angle(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_mag_angle

    convert a tensor containing rows of complex coefficient vectors into rows of real coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p complex tensor where p is the number modes

    Returns:
        torch.Tensor -- m by 2p tensor of elements in real numbers
    """
    if not torch.is_complex(coeff):
        logger.debug("coeff is already real")
        return coeff
    converted_coeff = torch.empty(
        (coeff.shape[0], coeff.shape[1] * 2), dtype=coeff.real.dtype
    )
    converted_coeff[:, ::2] = coeff.abs()
    converted_coeff[:, 1::2] = coeff.angle()

    return converted_coeff


def to_real_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_real_coeff

    convert a tensor containing rows of complex coefficient vectors into rows of real coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p complex tensor where p is the number modes

    Returns:
        torch.Tensor -- m by 2p tensor of elements in real numbers
    """
    if not torch.is_complex(coeff):
        logger.debug("coeff is already real")
        return coeff
    converted_coeff = torch.view_as_real(coeff).flatten(-2)

    return converted_coeff


class StandardScaler:
    def __init__(self):
        pass

    @staticmethod
    def _get_tensor_tuple(xs: tuple[torch.Tensor, ...] | torch.Tensor):
        if isinstance(xs, torch.Tensor):
            return (xs,)
        else:
            return xs

    @staticmethod
    def _get_m(x: torch.Tensor):
        x_real = to_real_coeff(x) if torch.is_complex(x) else x
        m = x_real.mean(0, keepdim=True)
        return m

    @staticmethod
    def _get_s(x: torch.Tensor, eps: float):
        x_real = to_real_coeff(x) if torch.is_complex(x) else x
        s = x_real.std(0, unbiased=False, keepdim=True)
        s[s <= eps] = eps
        return s

    def fit(self, xs: tuple[torch.Tensor, ...] | torch.Tensor, eps=1e-12):
        xs = self._get_tensor_tuple(xs)
        self.ms = tuple(self._get_m(x) for x in xs)
        self.ss = tuple(self._get_s(x, eps) for x in xs)
        self.xs_is_complex = tuple(torch.is_complex(x) for x in xs)
        self.xs_dims = tuple(x.shape[1] for x in xs)
        return self

    @staticmethod
    def _translate(x: torch.Tensor, m: torch.Tensor, i: int | None = None):
        x_name = "x" if i is None else f"xs[{i}]"
        assert x.shape[1] == m.shape[1], (
            f"{x_name} needs to have the same 2nd dimension now and at fitting"
        )
        x_translated = x.subtract(m)
        return x_translated

    @staticmethod
    def _scale(x: torch.Tensor, s: torch.Tensor, i: int | None = None):
        x_name = "x" if i is None else f"xs[{i}]"
        assert x.shape[1] == s.shape[1], (
            f"{x_name} needs to have the same 2nd dimension now and at fitting"
        )
        x_scaled = x.div(s)
        return x_scaled

    def _check_consistency(self, xs: tuple[torch.Tensor, ...]):
        xs_is_complex = tuple(torch.is_complex(x) for x in xs)
        assert xs_is_complex == self.xs_is_complex, (
            f"make sure the complex tensors are the same order when fitting, current complex: {xs_is_complex}, fitting complex: {self.xs_is_complex}"
        )
        xs_dims = tuple(x.shape[1] for x in xs)
        assert xs_dims == self.xs_dims, (
            f"make sure the tensors have the same 2nd dimensions with the ones used at fitting, current dims: {xs_dims}, fitting dims: {self.xs_dims}"
        )

    def get_subset_scaler(self, indices: int | list[int]):
        if isinstance(indices, int):
            indices = [indices]
        new_scaler = StandardScaler()
        new_scaler.ms = tuple(self.ms[index] for index in indices)
        new_scaler.ss = tuple(self.ss[index] for index in indices)
        new_scaler.xs_is_complex = tuple(self.xs_is_complex[index] for index in indices)
        new_scaler.xs_dims = tuple(self.xs_dims[index] for index in indices)
        return new_scaler

    @typing.overload
    def transform(self, xs: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def transform(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    def transform(self, xs: tuple[torch.Tensor, ...] | torch.Tensor):
        is_tensor_input = isinstance(xs, torch.Tensor)
        _xs = self._get_tensor_tuple(xs)
        self._check_consistency(_xs)
        xs_real = tuple(
            to_real_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, _xs, strict=False)
        )
        xs_transformed = tuple(
            self._translate(x, m, i=i)
            for i, (x, m) in enumerate(zip(xs_real, self.ms, strict=False))
        )
        xs_transformed = tuple(
            self._scale(x, s, i=i)
            for i, (x, s) in enumerate(zip(xs_transformed, self.ss, strict=False))
        )
        for x, m in zip(xs_transformed, self.ms, strict=False):
            for dim in range(x.shape[1]):
                x[:, dim].nan_to_num_(m[0, dim].item())

        xs_out = tuple(
            to_complex_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs_transformed, strict=False)
        )
        return xs_out[0] if is_tensor_input else xs_out

    @typing.overload
    def inverse(self, xs: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def inverse(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    def inverse(self, xs: tuple[torch.Tensor, ...] | torch.Tensor):
        is_tensor_input = isinstance(xs, torch.Tensor)
        _xs = self._get_tensor_tuple(xs)
        self._check_consistency(_xs)
        xs_real = tuple(
            to_real_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, _xs, strict=False)
        )
        xs_transformed = tuple(
            self._scale(x, 1.0 / s, i=i)
            for i, (x, s) in enumerate(zip(xs_real, self.ss, strict=False))
        )
        xs_transformed = tuple(
            self._translate(x, -1.0 * m, i=i)
            for i, (x, m) in enumerate(zip(xs_transformed, self.ms, strict=False))
        )
        for x, m in zip(xs_transformed, self.ms, strict=False):
            for dim in range(x.shape[1]):
                x[:, dim].nan_to_num_(m[0, dim].item())
        xs_out = tuple(
            to_complex_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs_transformed, strict=False)
        )
        return xs_out[0] if is_tensor_input else xs_out

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str):
        # weights_only=False: this pickles a full StandardScaler object, not a
        # state dict (torch>=2.6 defaults weights_only=True, which rejects it)
        scaler: StandardScaler = torch.load(path, weights_only=False)
        assert isinstance(scaler, StandardScaler), (
            "Loaded object is not a valid instance of StandardScaler"
        )
        return scaler


def scale_to_standard(x: torch.Tensor):
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled


MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _safe_r2(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # r2_score is undefined for a single sample.
    if targets.shape[0] > 1:
        return r2_score(preds, targets)
    return torch.tensor(float("nan"))


def _relative_squared_error(
    preds: torch.Tensor, targets: torch.Tensor, squared: bool = True
) -> torch.Tensor:
    """Global relative squared error: SSE / SST over ALL elements.

    ``rse = sum((p - t)^2) / sum((t - mean(t))^2)`` with a single global target
    mean; ``rrse = sqrt(rse)``. Deliberately NOT torchmetrics'
    ``relative_squared_error``, which averages per-column SSE/SST ratios -- a
    single near-constant target column then dominates the result (observed as
    rrse ~1e5 alongside r2 ~0.99 in the operator benchmark). The global form is
    consistent with the flattened r2 reported here: rse = 1 - r2.
    """
    sse = (preds - targets).square().sum()
    sst = (targets - targets.mean()).square().sum()
    rse = sse / sst
    return rse.sqrt() if not squared else rse


# Default metric set; inject a different mapping into get_metrics to override.
# NOTE: rse/rrse are normalized by the (global) target variance, so they blow
# up when the targets have near-zero variance overall; prefer mse/rmse there.
DEFAULT_METRICS: dict[str, MetricFn] = {
    "mse": mean_squared_error,
    "rmse": partial(mean_squared_error, squared=False),
    "mae": mean_absolute_error,
    "r2": _safe_r2,
    "smape": symmetric_mean_absolute_percentage_error,
    "rse": _relative_squared_error,
    "rrse": partial(_relative_squared_error, squared=False),
}


class Metrics(TypedDict):
    """Return type of ``get_metrics`` with the default metric set."""

    mse: float
    rmse: float
    mae: float
    r2: float
    smape: float
    rse: float
    rrse: float
    pred_nan_sum: int


@overload
def get_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Metrics: ...
@overload
def get_metrics(
    preds: torch.Tensor, targets: torch.Tensor, metrics: dict[str, MetricFn]
) -> dict[str, float]: ...


def get_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metrics: dict[str, MetricFn] | None = None,
) -> Metrics | dict[str, float]:
    active = DEFAULT_METRICS if metrics is None else metrics
    results = {name: fn(preds, targets).item() for name, fn in active.items()}
    results["pred_nan_sum"] = int(torch.isnan(preds).sum().item())
    # Default set matches the Metrics keys exactly; a custom set does not.
    return cast(Metrics, results) if metrics is None else results


def resize_modes(x: torch.Tensor, target_modes: int | tuple[int, ...], rescale=True):
    """Resize a coefficient grid to a different mode count per axis.

    ``rescale`` selects which of two quantities survives, since a change of
    mode count cannot preserve both. ``True`` preserves the represented
    function, multiplying by ``prod(target_modes) / prod(current_modes)`` to
    compensate the transform normalisation. ``False`` preserves the stored
    coefficient values, which scales the function by the mode-count ratio
    raised to the number of axes -- ``0.2500`` for a doubling in two
    dimensions, ``0.0625`` for a quadrupling.
    """
    current_modes = tuple(x.shape[1:])
    ncurrent_modes = len(current_modes)
    if isinstance(target_modes, int):
        target_modes = (target_modes,) * ncurrent_modes
    assert ncurrent_modes == len(target_modes), (
        f"x and max_modes should be the same dimensions after the first dimension of x, x has shape {x.shape} and max_modes is {target_modes}"
    )
    x_resized = x
    for dim, (target_mode, current_mode) in enumerate(
        zip(target_modes, current_modes, strict=False), 1
    ):
        device = x.device
        if target_mode < current_mode:
            start_range = torch.tensor(
                range((target_mode - 1) // 2 + 1), dtype=torch.int
            ).to(device=device)
            end_range = torch.tensor(
                range(current_mode - target_mode // 2, current_mode), dtype=torch.int
            ).to(device=device)

            x_resized = torch.concat(
                (
                    x_resized.index_select(dim, start_range),
                    x_resized.index_select(dim, end_range),
                ),
                dim,
            )
        elif target_mode > current_mode:
            start_range = torch.tensor(
                range((current_mode - 1) // 2 + 1), dtype=torch.int
            ).to(device=device)
            # high (negative-frequency) block: the last floor(N/2) modes, i.e.
            # indices [ceil(N/2), N). Start = N - N//2 = ceil(N/2). The old start
            # N//2 double-counts the middle mode for odd N (wrong output length).
            # Empty if the coefficient is size 1.
            high_start = current_mode - current_mode // 2  # = ceil(N/2)
            end_range = torch.tensor(
                range(high_start, current_mode) if current_mode > 1 else range(0),
                dtype=torch.int,
            ).to(device=device)
            padding_size = target_mode - current_mode
            modes = list(x_resized.shape)
            modes[dim] = padding_size
            padding = torch.zeros(modes).to(x_resized)

            x_resized = torch.concat(
                (
                    x_resized.index_select(dim, start_range),
                    padding,
                    x_resized.index_select(dim, end_range),
                ),
                dim,
            )

    if rescale:
        modes = list(x.shape[1:])
        x_resized = (
            x_resized * torch.tensor(target_modes).prod() / torch.tensor(modes).prod()
        )

    return x_resized


def zero_coeff(x: torch.Tensor, zeroed_modes: int | list[int]):
    """Band-limit a coefficient grid in place, keeping the lowest modes.

    An even count returns a grid that does not describe a real field. Modes
    pair as ``+-k`` around the lone mode ``0``, so a kept set whose inverse
    transform is real is always odd-sized: keeping an even count necessarily
    keeps one member of a pair without the other. No substitute result answers
    the request that was made -- rounding either direction keeps a different
    number of modes than asked for.
    """
    if isinstance(zeroed_modes, int):
        zeroed_modes = [zeroed_modes] * len(x.shape[1:])
    assert len(x.shape[1:]) == len(zeroed_modes), (
        f"x and max_modes should be the same dimensions after the first dimension of x, x has shape {x.shape} and max_modes is {zeroed_modes}"
    )
    x_zeroed = x.clone()
    for dim, max_mode in enumerate(zeroed_modes, 1):
        dim_len = x.shape[dim]
        zero_range = torch.tensor(
            range((max_mode - 1) // 2 + 1, dim_len - max_mode // 2)
        )
        x_zeroed.index_fill_(dim, zero_range, torch.zeros((1,), dtype=x.dtype)[0])

    return x_zeroed


def interpolate_tensor(x: torch.Tensor, index_float: torch.Tensor, dim: int = 1):
    index_floor = index_float.floor().to(torch.int)
    index_ceil = index_float.ceil().to(torch.int)
    if index_float.remainder(1).eq(0).all():
        x_interp = x.index_select(dim, index_floor)
    else:
        x_ceil = x.index_select(dim, index_ceil)
        x_floor = x.index_select(dim, index_floor)
        # interpolate coefficients
        index_shape = [1 for _ in range(x_floor.ndim)]
        index_shape[1] = -1
        index_scaler = (
            ((index_float - index_floor) / (index_ceil - index_floor))
            .reshape(index_shape)
            .nan_to_num()
        )
        # ynt + scaler * (ynt1 - ynt)
        # (1 - scaler) * ynt + scaler * ynt1
        x_interp = torch.lerp(x_floor, x_ceil, index_scaler.to(x))

    return x_interp


# SOLVERS

# Monolithic rhs(t, y) and the signature of a solver that integrates it. Kept
# for pluggable numerical solvers (e.g. torchdiffeq-backed) on future PDEs.
RHSFuncType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
SolverSignatureType = Callable[[RHSFuncType, torch.Tensor, torch.Tensor], torch.Tensor]
MixedRHSFuncType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MixedSolverSignatureType = Callable[
    [MixedRHSFuncType, torch.Tensor, torch.Tensor], torch.Tensor
]


def euler_solver(
    rhs_func: RHSFuncType,
    y0: torch.Tensor,
    t: torch.Tensor,
):
    assert len(t.shape) == 1, (
        "t should be a one dimensional tensor of all time evaluation points"
    )
    assert len(t) > 1, "t should have more than one evaluation points"

    solution = torch.zeros((len(t), *y0.shape)).to(y0)

    j = 1
    solution[j - 1] = y0
    for t0, t1 in zip(t[:-1], t[1:], strict=False):
        dt = t1 - t0
        y = solution[j - 1]
        solution[j] = y + dt * rhs_func(t0, y)
        assert solution.isnan().sum() == 0, (
            f"solver encountered nan at timestep {j} (t={t0})"
        )
        # print(f"j {j}")
        j = j + 1
    # print(f"j_last {j}")
    return solution


# Nonlinear term N(t, v) for a semilinear system dv/dt = L v + N(t, v).
SemilinearNonlinearType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
SemilinearSolverSignatureType = Callable[
    [torch.Tensor, SemilinearNonlinearType, torch.Tensor, torch.Tensor], torch.Tensor
]

ExpIntNonlinearType = SemilinearNonlinearType
ExpIntSolverSignatureType = SemilinearSolverSignatureType


def as_semilinear_solver(solver: SolverSignatureType) -> SemilinearSolverSignatureType:
    """Run a monolithic ``rhs(t, y)`` solver on a semilinear system.

    Forms ``rhs(t, v) = linear * v + nonlinear(t, v)`` and hands it to
    ``solver`` with the caller's ``y0`` and ``t`` unchanged -- no substituted
    step size or tolerance -- so the result is exactly what ``solver`` returns
    for the combined right-hand side.

    There is no conversion in the other direction: ``linear`` cannot be
    recovered from a combined right-hand side, which is why an exponential
    integrator such as :func:`etdrk4_solver` needs the split form.

    Accuracy: a solver designed for non-stiff systems loses accuracy on a stiff
    spectral operator, where the stiff eigenvalue magnitude grows as
    ``nu * k_max**2``. Agreement with an exponential integrator holds only while
    the step size resolves it, i.e. while ``h * nu * k_max**2`` is of order one
    or below; past that the wrapped solver degrades (torchdiffeq reports
    ``Functional iteration did not converge``) and this function does not hide
    it by adjusting the time grid.

    Arguments:
        solver {SolverSignatureType} -- integrator taking a single ``rhs(t, y)``.

    Returns:
        SemilinearSolverSignatureType -- the same integrator, callable as
        ``(linear, nonlinear, y0, t)``.
    """

    def solve(
        linear: torch.Tensor,
        nonlinear: SemilinearNonlinearType,
        y0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        def rhs(ti: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return linear * v + nonlinear(ti, v)

        return solver(rhs, y0, t)

    return solve


# ETDRK4 scheme and the complex contour-integral evaluation of its weight
# functions follow:
#   - A.-K. Kassam & L. N. Trefethen, "Fourth-order time-stepping for stiff
#     PDEs", SIAM J. Sci. Comput. 26(4), 2005.
#   - S. M. Cox & P. C. Matthews, "Exponential time differencing for stiff
#     systems", J. Comput. Phys. 176(2), 2002.
# Reference implementations / benchmarks consulted:
#   - https://github.com/gurcani/etdrk4cp
#   - https://github.com/whalenpt/rkstiff
#   - https://docs.sciml.ai/SciMLBenchmarksOutput/dev/SimpleHandwrittenPDE/burgers_spectral_wpd/
#   - ETDRK4 review preprint (preprints.org), provided by the project owner.


HALF_CIRCLE = torch.pi
FULL_CIRCLE = 2 * torch.pi


def _contour_nodes(count: int, arc: float, device: torch.device) -> torch.Tensor:
    j = torch.arange(1, count + 1, device=device, dtype=torch.float64)
    return torch.exp(1j * arc * (j - 0.5) / count)


def _etdrk4_weights(
    lin: torch.Tensor, h: float, contour_points: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contour-averaged ``hQ, hf1, hf2, hf3`` for ``complex128`` eigenvalues ``lin``.

    Real eigenvalues: ``contour_points`` nodes on the upper half circle, real
    part of the mean. Complex eigenvalues: twice as many nodes on the full
    circle, complex mean. Weights are shaped like ``lin``, real or complex
    accordingly.
    """
    shape = lin.shape
    if lin.imag.any():
        nodes = _contour_nodes(2 * contour_points, FULL_CIRCLE, lin.device)

        def average(expr: torch.Tensor) -> torch.Tensor:
            return (h * expr.mean(dim=-1)).reshape(shape)

    else:
        nodes = _contour_nodes(contour_points, HALF_CIRCLE, lin.device)

        def average(expr: torch.Tensor) -> torch.Tensor:
            return (h * expr.mean(dim=-1).real).reshape(shape)

    z = h * lin.reshape(-1).unsqueeze(-1) + nodes.unsqueeze(0)
    ez = torch.exp(z)
    Q = average((torch.exp(z / 2) - 1) / z)
    f1 = average((-4 - z + ez * (4 - 3 * z + z**2)) / z**3)
    f2 = average((2 + z + ez * (-2 + z)) / z**3)
    f3 = average((-4 - 3 * z - z**2 + ez * (4 - z)) / z**3)
    return Q, f1, f2, f3


def etdrk4_solver(
    linear: torch.Tensor,
    nonlinear: SemilinearNonlinearType,
    y0: torch.Tensor,
    t: torch.Tensor,
    contour_points: int = 32,
) -> torch.Tensor:
    """ETDRK4 for ``dv/dt = L v + N(t, v)`` with diagonal ``L``.

    Arguments:
        linear {torch.Tensor} -- (*modes) eigenvalues of ``L``, real or complex.
        nonlinear {Callable[[t, v], Nv]} -- ``N(t, v)`` on ``(samples, *modes)``,
            called at the stage times ``t_n, t_n + h/2, t_n + h/2, t_n + h``.
        y0 {torch.Tensor} -- (samples, *modes) initial state.
        t {torch.Tensor} -- (steps,) uniformly spaced times; ``h = t[1] - t[0]``.

    Keyword Arguments:
        contour_points {int} -- nodes for the weight contour integral; see
            :func:`_etdrk4_weights` (default: {32}).

    Returns:
        torch.Tensor -- (steps, samples, *modes).
    """
    assert t.ndim == 1 and len(t) > 1, "t must be a 1D tensor with >1 points"
    steps = t.diff()
    grid_rounding = 8 * torch.finfo(t.dtype).eps * t.abs().max()
    assert (steps - steps[0]).abs().max() <= grid_rounding, (
        "t must be uniformly spaced; ETDRK4 precomputes its weights from one step h"
    )
    assert y0.shape[1:] == linear.shape, (
        f"linear {tuple(linear.shape)} must match y0 mode grid {tuple(y0.shape[1:])}"
    )
    device = y0.device
    h = (t[1] - t[0]).item()

    lin = linear.to(device=device, dtype=torch.complex128)
    E = torch.exp(h * lin)
    E2 = torch.exp(h * lin / 2)

    Q, f1, f2, f3 = _etdrk4_weights(lin, h, contour_points)

    cdtype = y0.dtype if y0.is_complex() else torch.complex64
    rdtype = torch.tensor([], dtype=cdtype).real.dtype
    weight_dtype = cdtype if Q.is_complex() else rdtype
    E, E2 = E.to(cdtype), E2.to(cdtype)
    Q, f1, f2, f3 = (c.to(weight_dtype) for c in (Q, f1, f2, f3))

    v = y0.to(cdtype)
    solution = torch.empty((len(t), *y0.shape), dtype=cdtype, device=device)
    solution[0] = v
    for i in range(1, len(t)):
        t_start = t[i - 1]
        t_end = t[i]
        t_mid = (t_start + t_end) / 2
        nv = nonlinear(t_start, v)
        a = E2 * v + Q * nv
        na = nonlinear(t_mid, a)
        b = E2 * v + Q * na
        nb = nonlinear(t_mid, b)
        c = E2 * a + Q * (2 * nb - nv)
        nc = nonlinear(t_end, c)
        v = E * v + nv * f1 + 2 * (na + nb) * f2 + nc * f3
        solution[i] = v
    return solution


# torchdiffeq-backed ODE solvers matching SolverSignatureType, for pluggable
# numerical integration of (non-stiff) PDEs reduced to systems of ODEs.
implicit_adams_solver: SolverSignatureType = partial(
    odeint, method="implicit_adams", options={"max_iters": 4}
)

lsoda_solver: SolverSignatureType = partial(
    odeint, method="scipy_solver", options={"solver": "LSODA"}
)
