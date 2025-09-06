from .__base import (
    Basis,
    EvaluationModeType,
    PeriodsInputType,
    periodsInputType_to_tuple,
    ResType,
    transformResType_to_tuple,
)
from ..utils import to_complex_coeff
import torch
from typing_extensions import Self, Literal, Callable, overload
from functools import partial


## Fourier basis
class FourierBasis(Basis):
    coeff_dtype = torch.complex64

    def __init__(
        self,
        coeff: torch.Tensor | None = None,
        complex_funcs: bool = False,
        periods: PeriodsInputType = 1,
        time_dependent: bool = False,
    ) -> None:
        super().__init__(
            coeff,
            complex_funcs=complex_funcs,
            time_dependent=time_dependent,
            periods=periods,
        )

    @Basis.coeff.setter
    def coeff(self, coeff: torch.Tensor | None):
        if coeff is not None:
            assert coeff.is_complex(), "the coefficients passed in need to be complex"
        Basis.coeff.__set__(self, coeff)

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        i=0,
        n=0,
        periods: PeriodsInputType = None,
    ) -> torch.Tensor:
        coeff = self.coeff
        assert coeff is not None, (
            "coeff is none, set it in the function parameters or with setCoeff"
        )
        if periods is None:
            periods = self.periods
        modes = self.get_modes(coeff, time_dependent=self.time_dependent)
        assert modes is not None, (
            "modes is none, set it in the function parameters, at initialization of this basis, or via class properties"
        )
        return self.evaluate(
            coeff=coeff.to(device=x.device),
            x=x,
            t=t,
            periods=periods,
            i=i,
            n=n,
            time_dependent=self.time_dependent,
        )

    @overload
    @classmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        i=0,
        n=0,
        time_dependent: Literal[True] | bool = True,
        periods: PeriodsInputType = None,
    ) -> torch.Tensor: ...

    @overload
    @classmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        t: None = None,
        i=0,
        n=0,
        time_dependent: Literal[False] | bool = False,
        periods: PeriodsInputType = None,
    ) -> torch.Tensor: ...

    @classmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        i=0,
        n=0,
        time_dependent: bool = False,
        periods: PeriodsInputType = None,
    ) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        if n > 0:
            init_len = len(coeff)
            coeff = coeff[i : i + n]
            assert True if init_len < 1 else len(coeff) > 0, (
                "pass valid values for i and n"
            )
        if not coeff.is_complex():
            coeff = to_complex_coeff(coeff)

        modes = cls.get_modes(coeff, time_dependent=time_dependent)
        if time_dependent:
            assert t is not None, "t must not be none for time dependent evaluations"
            periods = periodsInputType_to_tuple(periods, (coeff.shape[1:]))
            assert len(periods) > 1, (
                f"periods given for time dependent evaluation must at least be of dimension 2, got {len(periods)}"
            )
            basis = cls.fn(
                x,
                modes,
                periods=periods[1:]
                if time_dependent and periods is not None
                else periods,
            )
            # evaluate
            sum_coeff_x_basis = cls.sum_mul(coeff.flatten(2), basis)
            # interpolate
            index_float = t.flatten().real / periods[0] * (coeff.shape[1] - 1)
            sum_coeff_x_basis = cls.interpolate_time_tensor(
                sum_coeff_x_basis, index_float
            )

        else:
            basis = cls.fn(x, modes, periods=periods)
            sum_coeff_x_basis = cls.sum_mul(coeff.flatten(1), basis)

        scaling = 1.0 / torch.prod(torch.Tensor(modes))
        return scaling * sum_coeff_x_basis

    @staticmethod
    def sum_mul(coeff_flat: torch.Tensor, basis: torch.Tensor):
        sum_coeff_x_basis = coeff_flat.matmul(basis.movedim(0, -1).flatten(0, -2))
        return sum_coeff_x_basis

    @staticmethod
    def fn(
        x: torch.Tensor,
        modes: int | tuple[int, ...],
        periods: PeriodsInputType | None = None,
        constant=2j * torch.pi,
        transpose: bool = False,
    ) -> torch.Tensor:
        if isinstance(modes, int):
            modes = (modes,)
        periods = periodsInputType_to_tuple(periods, modes)

        assert len(x.shape) > 1, (
            "x must have at least 2 dimensions, the format needs to be row of points, the first dimension of the tensor being each row and the second being dimensions of the points"
        )
        assert x.shape[0] > 0, (
            f"x has shape {x.shape}, make sure the first dimension isn't empty ie. has at least one row of samples"
        )
        assert x.shape[1] == len(modes), (
            f"x has dimensions {x.shape[1]} and modes has dimensions {len(modes)}, both need to have the same dimensions (modes specify how many modes in each dimension of the fourier series)"
        )
        assert x.shape[1] == len(periods), (
            f"x has dimensions {x.shape[1]} and periods has dimensions {len(periods)}, both need to have the same dimensions (periods the function periodicity in each dimension)"
        )
        ndims = x.shape[1]
        if not x.is_floating_point() and not x.is_complex():
            x = x.float()

        # Compute the Fourier basis functions
        # one time for each dimension

        dim = 0
        num_modes = modes[dim]
        dim_basis_shape = [1 for i in range(ndims + 1)]
        dim_basis_shape[0] = x.shape[0]
        dim_basis_shape[dim + 1] = num_modes
        kx = torch.zeros((x.shape[0], modes[0])).reshape(dim_basis_shape).to(x)

        for dim, num_modes in enumerate(modes):
            k = FourierBasis.wave_number(num_modes, target=x).T
            dim_basis_shape = [1 for i in range(ndims + 1)]
            dim_basis_shape[0] = x.shape[0]
            dim_basis_shape[dim + 1] = num_modes
            dim_x = x[:, dim : dim + 1].div(periods[dim])
            dim_kx = torch.mm(dim_x, k)
            if transpose:
                dim_kx = dim_kx.T
            dim_kx = dim_kx.reshape(dim_basis_shape)

            kx = kx + dim_kx
        kx = kx.mul(constant).exp_()
        basis = kx

        return basis

    @staticmethod
    def wave_number(modes: int, target: torch.dtype | torch.Tensor = torch.float):
        N = (modes - 1) // 2 + 1
        n = modes // 2
        k1 = torch.arange(0, N)
        k2 = torch.arange(-n, 0)
        k = torch.concat([k1, k2], dim=0).unsqueeze(-1).to(target)
        return k

    @classmethod
    def generate(
        cls,
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        random_func=torch.randn,
        complex_funcs: bool = False,
        periods: PeriodsInputType = None,
        value_type: Literal["random", "zero"] = "random",
    ) -> Self:
        match value_type:
            case "random":
                coeff = cls.generate_coeff(
                    n,
                    modes,
                    generator=generator,
                    random_func=random_func,
                    complex_funcs=complex_funcs,
                )
            case "zero":
                coeff = cls.generate_empty(n, modes)
            case _:
                raise RuntimeError(
                    f"FourierBasis generate expects value_type random or zero but got {value_type}"
                )

        return cls(
            coeff,
            periods=periods,
            complex_funcs=complex_funcs,
        )

    @classmethod
    def generate_empty(
        cls,
        n: int,
        modes: int | tuple[int, ...],
    ) -> torch.Tensor:
        if isinstance(modes, int):
            modes = (modes,)
        n_modes = len(modes)
        assert n_modes > 0, "modes should have at least one dimension"
        coeff = torch.zeros((n, *modes), dtype=torch.complex64)
        return coeff

    @classmethod
    def generate_coeff(
        cls,
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        random_func: Callable[..., torch.Tensor] = torch.randn,
        complex_funcs: bool = False,
        scale: bool = True,
    ) -> torch.Tensor:
        if isinstance(modes, int):
            modes = (modes,)
        n_modes = len(modes)
        assert n_modes > 0, "modes should have at least one element"
        random_func = partial(random_func, generator=generator)
        if complex_funcs:
            coeff = random_func((n, *modes), dtype=torch.complex64)
        else:
            coeff = random_func((n, *modes), dtype=torch.complex64)
            # TODO: replace with more efficient algo
            # Analysis of fourier transform outputs
            # dims = 2
            # mode = 8
            # modes = (mode,) * dims
            # tmp = FourierBasis.generate(1, modes, complex_funcs=True)
            # tmp_coeff = resize_modes(
            #     FourierBasis.transform(
            #         tmp(FourierBasis.grid([slice(0, 1, 200), slice(0, 1, 200)]).flatten(0, -2))
            #         .real.add(0j)
            #         .reshape((1, 200, 200))
            #     ),
            #     modes,
            # )
            # def get_reflected_modes(tmp_coeff):
            #     from functools import reduce

            #     dims = torch.tensor(tmp_coeff.shape[1:])
            #     n = 0
            #     results = []
            #     for num in range(torch.prod(dims)):
            #         idx = tuple(
            #             reduce(
            #                 lambda a, b: a // b,
            #                 dims[i + 1 :].tolist() if i + 1 < len(dims) else [],
            #                 num,
            #             )
            #             % dims[i].item()
            #             for i in range(len(dims))
            #         )
            #         # for k in range(dims[2]):
            #         eq_conj = tmp_coeff[n].isclose(tmp_coeff[(n, *idx)].conj(), 0.001)
            #         # eq_conj = tmp_coeff[n].isclose(tmp_coeff[n, i, j, k].conj(), 0.001)
            #         num_eq = eq_conj.sum()
            #         results.append((idx, num_eq, eq_conj.nonzero()))
            #     return sum(map(lambda x: x[1].item(), results)), torch.prod(dims).item(), results

            # get_reflected_modes(tmp_coeff)

            vals = cls.inv_transform(coeff)
            coeff = cls.transform(vals.real + 0j)
        if scale:
            # TODO: fix this to be more exact
            scaler = torch.tensor(modes).sum() * 0.2
            coeff = coeff.mul(scaler)
            # pass
        return coeff

    @staticmethod
    def _raw_transform(
        f: torch.Tensor,
        func: Literal["forward", "inverse"],
        res: slice,
        periodic: bool,
        period: float,
        allow_fft: bool,
    ) -> torch.Tensor:
        assert torch.is_complex(f), (
            "f is not complex, cast it to complex first eg. f + 0j"
        )
        match func:
            case "forward":
                sign = -1
            case "inverse":
                sign = 1
        mode = f.shape[1]
        domain_starts_at_0 = res.start == 0
        domain_end_equal_to_period = res.stop == period
        can_use_fft = (
            domain_starts_at_0
            and domain_end_equal_to_period
            and periodic
            and allow_fft
            and mode
            == res.step  # TODO: using FFT can mess up the transform when evaluating at different resolutions
        )
        if can_use_fft:
            if func == "forward":
                F = torch.fft.fft(f, dim=1, n=res.step, norm="backward")
            elif func == "inverse":
                F = torch.fft.ifft(f, dim=1, n=res.step, norm="forward")
        else:
            if periodic:  # tn=t0, {t0,...,tn-1}
                n = torch.arange(
                    res.start, res.stop, (res.stop - res.start) / res.step
                ).to(f)
            else:
                n = torch.linspace(res.start, res.stop, res.step).to(f)
            e = FourierBasis.fn(
                n.view(-1, 1),
                mode,
                periods=period,
                constant=sign * 2j * torch.pi,
            )
            # TODO: fix performance problem with very narrow tensors (eg. size 1x400000 tensors)

            F = torch.mm(f, e.T)

        assert isinstance(F, torch.Tensor), (
            f"Something went wrong during the raw transform, expected result of {torch.Tensor}, but got {type(F)}"
        )

        return F

    @staticmethod
    def _ndim_transform(
        f: torch.Tensor,
        dim: int,
        func: Literal["forward", "inverse"],
        res: slice,
        periodic: bool,
        period: float,
        allow_fft: bool,
    ) -> torch.Tensor:
        # flatten so that each extra dimension is treated as a separate "sample"
        # move dimension to transform to the end so that it can stay intact after f is flatened
        f_transposed = f.moveaxis(dim, -1)
        # flatten so that the last dimension is intact
        f_flatened = f_transposed.flatten(0, -2)

        F_flattened = FourierBasis._raw_transform(
            f_flatened,
            func=func,
            res=res,
            periodic=periodic,
            period=period,
            allow_fft=allow_fft,
        )
        # unflatten so that the correct shape is returned
        F_transposed = F_flattened.reshape((*f_transposed.shape[:-1], res.step))
        F = F_transposed.moveaxis(-1, dim)

        return F

    @staticmethod
    def transform(
        f: torch.Tensor,
        res: ResType | None = None,
        # TODO: change this to false and adjust affected areas like the notebooks and Basis.get_value()
        # This should be false because all Basis transforms should have the same
        # default behavior, in this case is including the end of the grid (non periodicity)
        # periodicity introduces inconsistency in the assumed grid for other functions built on basis transforms
        periodic: bool = True,
        periods: PeriodsInputType = None,
        allow_fft: bool = True,
    ) -> torch.Tensor:
        """
        transform

        Function to calculate the
        discrete Fourier Transform
        of complex-valued signal f

        Arguments:
            f {torch.Tensor} -- m discretized real valued functions
            res {tuple[slice,...] | None} -- resolution the function was evaluated at and the bounds of the evaluation (dafault: {None}). When res is None, the evaluation takes the same resolution as f with bounds [0,period) if periodic or [0,period] if not periodic.
            periodic {bool} -- whether the evaluation grid should include the end or not (periodic) (default: {True})
            periods: {Number | list[Number] | tuple[Number, ...] | None} -- evaluation period (default: {1})
            allow_fft {bool} -- allow the use of torch.fft module (default: {True}). By default the function will use fft if possible (domain is [0,1) which is also periodic)

        Returns:
            torch.Tensor -- m complex valued coefficients of f
        """
        ndims = len(f.shape)
        assert ndims >= 2, (
            f"f has shape {f.shape}, It needs to have at least two dimensions with the first being m samples"
        )
        if not torch.is_complex(f):
            f = f * (1 + 0j)
        periods = periodsInputType_to_tuple(periods, f.shape[1:])
        # Res should by default evaluate to the period not 1
        res = transformResType_to_tuple(res, tuple(f.shape[1:]), periods)
        # perform 1d transform over every dimension
        F = f
        for cdim in range(1, ndims):
            F = FourierBasis._ndim_transform(
                F,
                dim=cdim,
                func="forward",
                res=res[cdim - 1],
                periodic=periodic,
                period=periods[cdim - 1],
                allow_fft=allow_fft,
            )

        return F

    @staticmethod
    def inv_transform(
        F: torch.Tensor,
        res: ResType | None = None,
        periodic: bool = True,
        scale: bool = True,
        periods: PeriodsInputType = None,
        allow_fft: bool = True,
    ):
        """
        inv_transform

        Function to calculate the
        discrete Inverse Fourier Transform
        of coefficients F

        Arguments:
            f {torch.Tensor} -- m discretized complex valued coefficients with K modes
            res {tuple[slice,...] | None} -- resolution to evaluate the function at and the bounds of the evaluation (dafault: {None}). When res is None, the evaluation takes the same resolution as f with bounds [0,period) if periodic or [0,period] if not periodic.
            periodic {bool} -- whether the evaluation grid should include the end or not (periodic) (default: {True})
            scale {bool} -- whether the outputs are scaled by N or not (default: {True})
            periods {Number | list[Number] | tuple[Number, ...] | None} -- evaluation period (default: {1})
            allow_fft {bool} -- allow the use of torch.fft module (default: {True}). By default the function will use fft if possible (domain is [0,1) which is also periodic)

        Returns:
            torch.Tensor -- m complex valued coefficients of f
        """
        ndims = len(F.shape)
        assert ndims >= 2, (
            f"f has shape {F.shape}, It needs to have at least two dimensions with the first being m samples"
        )
        if not torch.is_complex(F):
            F = F * (1 + 0j)
        periods = periodsInputType_to_tuple(periods, F.shape[1:])
        # Res should by default evaluate to the period not 1
        res = transformResType_to_tuple(res, tuple(F.shape[1:]), periods)

        # perform 1d transform over every dimension
        f = F
        for cdim in range(1, ndims):
            f = FourierBasis._ndim_transform(
                f,
                dim=cdim,
                func="inverse",
                res=res[cdim - 1],
                periodic=periodic,
                period=periods[cdim - 1],
                allow_fft=allow_fft,
            )

        if scale:
            f = f.div(torch.tensor(F.shape[1:]).prod())
        return f

    def grad(self, dim: int = 0, ord: int = 1) -> Self:
        copy = self.copy()
        if dim == 0 and self.time_dependent:
            # time dependent use finite differences
            dt = self.periods[0] / (self.time_size - 1)
            coeff = copy.coeff
            for _ in range(ord):
                coeff = torch.gradient(coeff, spacing=dt, dim=1)[0]
            copy.coeff = coeff
        else:
            if self.time_dependent:
                # disregard time dimension
                dim = dim - 1
            k = copy.wave_number(copy.modes[dim])
            multiplier_dims = [1 for _ in range(copy.ndim)]
            multiplier_dims[dim] = copy.modes[dim]
            if self.time_dependent:
                multiplier_dims = (1, *multiplier_dims)
            multiplier = (
                2
                * torch.pi
                * 1j
                * k.reshape(multiplier_dims).to(copy.coeff)
                / self.periods[dim]
            )
            multiplier = multiplier.pow(ord)
            coeff = copy.coeff.mul(multiplier)
            coeff[:, ..., 0] = torch.tensor(0 + 0j)
            copy.coeff = coeff
        return copy

    def integral(self, dim: int = 0, ord: int = 1) -> Self:
        copy = self.copy()
        if dim == 0 and self.time_dependent:
            # time dependent use finite differences
            dt = self.periods[0] / (self.time_size - 1)
            coeff = copy.coeff
            for _ in range(ord):
                coeff = coeff.cumsum(1).mul(dt)
            copy.coeff = coeff
        else:
            if self.time_dependent:
                # disregard time dimension
                dim = dim - 1
            k = copy.wave_number(copy.modes[dim])
            multiplier_dims = [1 for _ in range(copy.ndim)]
            multiplier_dims[dim] = copy.modes[dim]
            if self.time_dependent:
                multiplier_dims = (1, *multiplier_dims)
            multiplier = (
                2
                * torch.pi
                * 1j
                * k.reshape(multiplier_dims).to(copy.coeff)
                / self.periods[dim]
            )
            multiplier = multiplier.pow(ord)
            coeff = copy.coeff.div(multiplier)
            coeff[:, ..., 0] = torch.tensor(0 + 0j)
            copy.coeff = coeff
        return copy

    def copy(self) -> Self:
        basis_copy = super().copy()
        return basis_copy

    @staticmethod
    def prefered_evaluation_mode() -> EvaluationModeType:
        return "inverse transform"
