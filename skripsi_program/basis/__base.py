import torch
import abc
from typing_extensions import Self, Literal, TypeVar, overload
import matplotlib.pyplot as plt
import logging
from ..utils import resize_modes
from ..utils import Number
# Basis functions
# - able to set number of modes / basis functions
# - provides access to the vector of basis function values evaluated at x

logger = logging.getLogger(__name__)
ResType = int | slice | tuple[slice, ...]
EvaluationModeType = Literal["inverse transform", "basis"]
AutoEvaluationModeType = Literal["auto"] | EvaluationModeType
PeriodsInputType = Number | list[Number] | tuple[Number, ...] | None
TransformResType = int | tuple[slice, ...]


def periodsInputType_to_tuple(
    periods: PeriodsInputType, modes: tuple[int, ...]
) -> tuple[float, ...]:
    if isinstance(periods, list) or isinstance(periods, tuple):
        periods = tuple(float(period) for period in periods)
    else:
        if periods is None:
            periods = 1.0
        periods = float(periods)
        periods = tuple(periods for _ in range(len(modes)))
    return periods


def transformResType_to_tuple(
    res: TransformResType | None, modes: tuple[int, ...]
) -> tuple[slice, ...]:
    if res is None:
        _res = tuple(slice(0, 1, mode) for mode in modes)
    elif isinstance(res, int):
        _res = tuple(slice(0, 1, res) for mode in modes)
    else:
        assert len(res) == len(
            modes
        ), f"expected res of length {len(modes)} but got length {len(res)}"
        _res = res
    return _res


class Basis(abc.ABC):
    """
    Basis function for ndim dimensions
    """

    _coeff: torch.Tensor

    def __init__(
        self,
        coeff: torch.Tensor | None = None,
        periods: PeriodsInputType = 1,
        complex_funcs: bool = False,
        time_dependent: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.coeff = coeff
        self._complex_funcs = complex_funcs
        self.time_dependent = time_dependent
        self.periods = periods

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    @abc.abstractmethod
    def coeff(self, coeff: torch.Tensor | None):
        if coeff is None:
            self._coeff = torch.empty(0)
        else:
            assert (
                coeff.ndim >= 2
            ), "coeff needs to be at least a two dimensional tensor of coefficients"
            self._coeff = coeff

    @property
    def modes(self) -> tuple[int, ...]:
        if self.ndim < 1:
            return (0,)
        else:
            return self.get_modes(self.coeff, self.time_dependent)

    @staticmethod
    def get_modes(coeff: torch.Tensor, time_dependent: bool) -> tuple[int, ...]:
        if time_dependent:
            return tuple(coeff.shape[2:])
        else:
            return tuple(coeff.shape[1:])

    @property
    def ndim(self):
        coeff_ndim = self.coeff.ndim
        if self.coeff is None or coeff_ndim < 1:
            return 0
        if self.time_dependent:
            return coeff_ndim - 2
        else:
            return coeff_ndim - 1

    @property
    def time_size(self):
        if not self.time_dependent:
            return 0
        return self.coeff.shape[1]

    @property
    def periods(self) -> tuple[float, ...]:
        return self._periods

    @periods.setter
    def periods(
        self,
        periods: PeriodsInputType,
    ):
        self._periods = periodsInputType_to_tuple(
            periods,
            (self.time_size, *self.modes) if self.time_dependent else self.modes,
        )

    @staticmethod
    @abc.abstractmethod
    def prefered_evaluation_mode() -> EvaluationModeType:
        """
        prefered_evaluation_mode

        returns the prefered evaluation mode of a basis

        Returns:
            EvaluationModeType -- the prefered evaluation mode
        """
        ...

    def get_values_and_grid(
        self,
        i=0,
        n=-1,
        res: ResType | None = None,
        evaluation_mode: AutoEvaluationModeType = "auto",
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        get_values_and_grid

        computes the values and evaluation grid of the coefficients

        Keyword Arguments:
            i {int} -- function i to start plotting (default: {0})
            n {int} -- n functions after function i to evaluate (default: {-1}). The default evaluates all functions
            res {int | slice | list[slice] | None} -- function discretization resolution and domain (default: {0:1:200} domain from 0 to one on all dimensions with 5 times the number of modes in points each). By default if only an int or a single slice is given, every dimension will share the same range and the resolution is based on the number of dimensions.
            evaluation_mode {"auto" | "inverse transform" | "basis"} -- coefficient evaluation mode (default: {"auto"}). Auto will use the inverse transform if the number of evaluations is high or res is not provided
            device {torch.device | None} -- device the evaluations are done on (default: {None}). By default, the function will try to use the GPU and fallback on the CPU.

        Returns:
            tuple[torch.Tensor, torch.Tensor] -- tuple of value and grid respectively of the evaluated functions
        """
        if device is None:
            device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        if evaluation_mode == "auto":
            evaluation_mode = self.prefered_evaluation_mode()
        evaluation_dim = (self.ndim + 1) if self.time_dependent else self.ndim
        if res is None:
            res = slice(0, 1, 200)
        res_t = None
        if isinstance(res, int):
            res = (slice(0, 1, res),)
        elif isinstance(res, slice):
            res = (res,)

        if len(res) == 1:
            res = res * evaluation_dim

        if evaluation_mode == "inverse transform":
            if n > 0:
                coeff = self.coeff[i : i + n]
            else:
                coeff = self.coeff
            coeff_shape = coeff.shape
            coeff = coeff.to(device=device)

            if self.time_dependent:
                res_t = res[0]
                res_spatial = res[1:]
                t = self.grid(res_t).to(device=device)
                index_float = t.flatten() / self.periods[0] * (coeff.shape[1] - 1)
                index_floor = index_float.floor().to(torch.int)
                index_ceil = index_float.ceil().to(torch.int)
                coeff_ceil = coeff[:, index_ceil].reshape((-1, *self.modes))
                coeff_floor = coeff[:, index_floor].reshape((-1, *self.modes))
                # evaluate
                values_ceil = self.inv_transform(coeff_ceil, res=res_spatial)
                values_floor = self.inv_transform(coeff_floor, res=res_spatial)
                values_shape = (coeff_shape[0], -1, *values_ceil.shape[1:])
                values_ceil = values_ceil.reshape(values_shape)
                values_floor = values_floor.reshape(values_shape)
                # interpolate time values
                index_shape = [1 for _ in range(values_ceil.ndim)]
                index_shape[1] = -1
                index_scaler = (
                    ((index_float - index_floor) / (index_ceil - index_floor))
                    .reshape(index_shape)
                    .nan_to_num()
                )
                values = values_floor.add(
                    (values_ceil - values_floor) * index_scaler
                ).to(self.coeff)
            else:
                res_spatial = res
                values = self.inv_transform(coeff, res=res_spatial).to(self.coeff)

            grid = self.grid(res)
        else:
            if self.time_dependent:
                assert (
                    len(res) > 1
                ), "res list should be more than one element for time dependent coefficients"
                res_t = res[0]
                res = res[1:]

            grid = self.grid(res)
            grid_device = grid.to(device=device)
            if res_t is None:
                grid_t = None
                grid_t_device = None
                grid_shape = tuple(r.step for r in res)
            else:
                grid_t = self.grid(res_t)
                grid_t_device = grid_t.to(device=device)
                grid_shape = tuple(r.step for r in (res_t, *res))

            values = self.__call__(
                grid_device.flatten(0, -2), t=grid_t_device, i=i, n=n
            ).reshape((-1, *grid_shape))
            if res_t is not None:
                # grid with the time coordinates for complete grid
                grid = self.grid((res_t, *res))

        return values, grid

    def get_values(
        self,
        i=0,
        n=-1,
        res: ResType | None = None,
        evaluation_mode: AutoEvaluationModeType = "auto",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        get_values

        computes the values of the coefficients

        Keyword Arguments:
            i {int} -- function i to start plotting (default: {0})
            n {int} -- n functions after function i to evaluate (default: {-1}). The default evaluates all functions
            res {int | slice | list[slice] | None} -- function discretization resolution and domain (default: {0:1:200} domain from 0 to one on all dimensions with 5 times the number of modes in points each). By default if only an int or a single slice is given, every dimension will share the same range and the resolution is based on the number of dimensions.
            evaluation_mode {"auto" | "inverse transform" | "basis"} -- coefficient evaluation mode (default: {"auto"}). Auto will use the inverse transform if the number of evaluations is high or res is not provided
            device {torch.device | None} -- device the evaluations are done on (default: {None}). By default, the function will try to use the GPU and fallback on the CPU.

        Returns:
            torch.Tensor -- value of the evaluated functions
        """
        return self.get_values_and_grid(
            i=i, n=n, res=res, evaluation_mode=evaluation_mode
        )[0]

    def __len__(self):
        if self.coeff is None:
            return 0
        return self.coeff.__len__()

    @staticmethod
    @abc.abstractmethod
    def fn(
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        fn

        evaluate the value of the basis functions

        Arguments:
            x {torch.Tensor} -- the m by ndim matrix of points to evaluate the basis functions at.

        Returns:
            torch.Tensor -- returns a vector using a tensor of the shape {m,modes}
        """
        pass

    @abc.abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        *args,
        t: torch.Tensor | None = None,
        i: int = 0,
        n: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        __call__

        evaluate approximated function at points x

        Arguments:
            x {torch.Tensor} -- m points to evaluate approximated function at

        Keyword Arguments:
            t {torch.Tensor | None} -- p time coordinates for time dependent coordinates (default: {None})
            i {int} -- function i to start evaluations at (default: {0})
            n {int} -- n functions after function i to evaluate (default: {0} all functions)

        Returns:
            torch.Tensor -- {n, p, m} evaluations where n is the number different functions (coeff first dimension)
        """
        pass

    @overload
    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        *args,
        t: torch.Tensor,
        i=0,
        n=0,
        time_dependent: Literal[True] | bool = True,
        **kwargs,
    ) -> torch.Tensor: ...

    @overload
    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        *args,
        t: None = None,
        i=0,
        n=0,
        time_dependent: Literal[False] | bool = False,
        **kwargs,
    ) -> torch.Tensor: ...

    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        coeff: torch.Tensor,
        x: torch.Tensor,
        *args,
        t: torch.Tensor | None = None,
        i: int = 0,
        n: int = 0,
        time_dependent: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        evaluate

        evaluate approximated function of coeff at points x

        Arguments:
            coeff {torch.Tensor} -- coefficients of approximated functions
            x {torch.Tensor} -- m points to evaluate approximated function at

        Keyword Arguments:
            t {torch.Tensor | None} -- p time coordinates for time dependent coefficients (default: {None})
            i {int} -- function i to start evaluations at (default: {0})
            n {int} -- n functions after function i to evaluate (default: {0} all functions)
            time_dependent {bool} -- whether the coefficients are time dependent or not (default: {False} not time dependent coefficients)

        Returns:
            torch.Tensor -- {n, m} evaluations where n is the number different functions (coeff first dimension)
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def transform(
        f: torch.Tensor, res: TransformResType | None = None, **kwargs
    ) -> torch.Tensor:
        """
        transform

        compute basis coefficients

        Arguments:
            f {torch.Tensor} -- m vectors of descretized functions to compute the coefficients of.
            res {tuple[slice,...] | None} -- resolution to evaluate the function at and the bounds of the evaluation (dafault: {None}). When res is None, the evaluation takes the same resolution as f with bounds [0,1).

        Returns:
            torch.Tensor -- m vectors of coefficients
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def inv_transform(
        f: torch.Tensor, res: TransformResType | None = None, **kwargs
    ) -> torch.Tensor:
        """
        inv_transform

        compute function values from dft coefficients

        Arguments:
            f {torch.Tensor} -- m vectors of coefficeints to compute the function values of.
            res {tuple[slice,...] | None} -- resolution to evaluate the coefficients at and the bounds of the evaluation (dafault: {None}). When res is None, the evaluation takes the same resolution as f with bounds [0,1).

        Returns:
            torch.Tensor -- m vectors of function values
        """
        pass

    @classmethod
    @abc.abstractmethod
    def generate(
        cls,
        n: int,
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        random_func=torch.randn,
        **kwargs,
    ) -> Self:
        """
        generate

        generate functions using basis functions with random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.
            modes {int | tuple[int,...]} -- number of coefficients in a series.

        Keyword Arguments:
            generator {torch.Generator | None} -- PRNG Generator for reproducability (default: {None})
            random_func {callable} -- random function that generates the coefficients (default: {torch.randn})

        Returns:
            Basis -- n sets of functions with coefficients with the shape (n, modes)
        """

    @classmethod
    @abc.abstractmethod
    def generate_coeff(
        cls, n: int, modes: int | tuple[int, ...], *args, **kwargs
    ) -> torch.Tensor:
        """
        generate_coeff

        generate random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.
            modes {int | tuple[int,...]} -- number of coefficients in each function.

        Returns:
            torch.Tensor -- n sets of coefficients with the shape (n, modes)
        """

    @classmethod
    @abc.abstractmethod
    def generate_empty(
        cls, n: int, modes: int | tuple[int, ...], *args, **kwargs
    ) -> torch.Tensor:
        """
        generate_empty

        generates coefficient array with zero values

        Arguments:
            n {int} -- number of random functions to generate coefficients for.
            modes {int | tuple[int, ...]} -- number of coefficients in each function

        Raises:
            TypeError: _description_
            TypeError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            torch.Tensor -- n sets of coefficients with the shape (n, modes)
        """

    @abc.abstractmethod
    def grad(self, dim: int = 0) -> Self:
        """
        grad

        grad computes the derivative along a dimension

        Keyword Arguments:
            dim {int} -- dimension to compute gradient on (default: {0})

        Returns:
            Self -- returns an instance of current basis with antiderivative coefficients
        """

    @abc.abstractmethod
    def integral(self, dim: int = 0) -> Self:
        """
        integral

        integral computes the antiderivative along a dimension

        Keyword Arguments:
            dim {int} -- dimension to compute antiderivative on (default: {0})

        Returns:
            Self -- returns an instance of current basis with antiderivative coefficients
        """

    @abc.abstractmethod
    def copy(self) -> Self:
        """
        copy

        copy returns an instance of the same basis subclass with identical attributes

        Returns:
            Self -- a copied instance of current instance
        """
        return self.__class__(coeff=self.coeff, complex_funcs=self._complex_funcs)

    def __sub__(self, other: Self):
        if isinstance(other, self.__class__):
            if other.coeff is None:
                return self.copy()
            elif self.coeff is None:
                other_copy = other.copy()
                other_copy.coeff = -other.coeff.clone()
                return other_copy
            else:
                result = self.resize_modes(other)
                result.coeff = result.coeff - other.coeff
                return result
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'"
            )

    def __add__(self, other: Self):
        if isinstance(other, self.__class__):
            if other.coeff is None:
                return self.copy()
            elif self.coeff is None:
                return other.copy()
            else:
                result = self.resize_modes(other)
                result.coeff = result.coeff + other.coeff
                return result
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'"
            )

    # TODO: Add plot coefficients function

    def plot(
        self,
        i=0,
        n=1,
        res: ResType | None = None,
        plt=plt,
        complex_scatter=False,
        plot_component: None | Literal["imag", "real"] = None,
        legend: bool = True,
        evaluation_mode: AutoEvaluationModeType = "auto",
        device: torch.device | None = None,
        **kwargs,
    ):
        """
        plot

        plot draws a plot of the functions in the basis, extra keyword arguments are passed to the matplotlib plotting functions.

        Keyword Arguments:
            i {int} -- function i to start plotting (default: {0})
            n {int} -- n functions after function i to evaluate (default: {1})
            res {int | slice | list[slice] | None} -- function discretization resolution and domain (default: {0:1:min(2x modes, 200)} domain from 0 to one on all dimensions with 5 times the number of modes in points each). By default if only an int or a single slice is given, every dimension will share the same range and the resolution is based on the number of modes. Currently dimension higher than 2 just uses the inverse transform and ignores this parameter.
            plt {_type_} -- Axes or pyplot to get plot or imshow from (default: {pyplot})
            complex_scatter {bool} -- plot the complex values on a scatter plot instead (default: {False})
            plot_component {None | Literal[&quot;imag&quot;, &quot;real&quot;]} -- plot the imaginary or real values or both if None (default: {None})
            legend {bool} -- add legend to plots (default: {True})
            evaluation_mode {"auto" | "inverse transform" | "basis"} -- coefficient evaluation mode (default: {"auto"}). Auto will use the inverse transform if the number of evaluations is high or res is not provided.
            device {torch.device | None} -- device the evaluations are done on (default: {None}). By default, the function will try to use the GPU and fallback on the CPU.

        Raises:
            NotImplementedError: TODO: implement dimension higher than 2
            NotImplementedError: For 2 dimension, no plotting is available, try using complex scatter instead

        Returns:
            _type_ -- returns the result of the plotting function such as list[Line2D] or AxesImage
        """
        plot_modes = (
            (self.time_size, *self.modes) if self.time_dependent else self.modes
        )
        if res is None:
            res = tuple(slice(0, 1, 200) for mode in plot_modes)
        plot_dims = self.ndim + 1 if self.time_dependent else self.ndim
        values, grid = self.get_values_and_grid(
            i=i, n=n, res=res, evaluation_mode=evaluation_mode
        )
        values = values.cpu()
        grid = grid.cpu()

        match plot_dims:
            case 1:
                if self._complex_funcs:
                    if complex_scatter:
                        for func in values:
                            func_flat = func.flatten()
                            plot = plt.scatter(
                                func_flat.real,
                                func_flat.imag,
                                **kwargs,
                            )
                        if legend:
                            plt.legend(
                                [f"Function ({i+j})" for j in range(len(values))]
                            )
                    else:
                        match plot_component:
                            case "real":
                                for func in values:
                                    func_flat = func.flatten()
                                    plot = plt.plot(
                                        grid.flatten(),
                                        func_flat.real,
                                        **kwargs,
                                    )
                                if legend:
                                    plt.legend(
                                        [
                                            f"Real function ({i+j})"
                                            for j in range(len(values))
                                        ]
                                    )
                            case "imag":
                                for func in values:
                                    func_flat = func.flatten()
                                    plot = plt.plot(
                                        grid.flatten(),
                                        func_flat.imag,
                                        linestyle="dashed",
                                        **kwargs,
                                    )
                                if legend:
                                    plt.legend(
                                        [
                                            f"Imaginary function ({i+j})"
                                            for j in range(len(values))
                                        ]
                                    )
                            case _:
                                for func in values:
                                    func_flat = func.flatten()
                                    plot = plt.plot(
                                        grid.flatten(),
                                        func_flat.real,
                                        **kwargs,
                                    )
                                    kwargs["color"] = kwargs.get(
                                        "color", plot[0].get_color()
                                    )
                                    kwargs["linestyle"] = kwargs.get(
                                        "linestyle", "dashed"
                                    )
                                    plot = plt.plot(
                                        grid.flatten(),
                                        func_flat.imag,
                                        **kwargs,
                                    )
                                if legend:
                                    plt.legend(
                                        [
                                            f"Real function ({i+j})"
                                            if k == 0
                                            else f"Imaginary function ({i+j})"
                                            for k in range(2)
                                            for j in range(len(values))
                                        ]
                                    )
                else:
                    for func in values:
                        plot = plt.plot(grid.flatten(), func.flatten().real, **kwargs)
                    if legend:
                        plt.legend(
                            [(f"Real function ({i+j})") for j in range(len(values))]
                        )
            case 2:
                if complex_scatter:
                    for func in values:
                        func_flat = func.flatten()
                        plot = plt.scatter(func_flat.real, func_flat.imag, **kwargs)
                    if legend:
                        plt.legend([f"Function ({i+j})" for j in range(len(values))])
                else:
                    if plot_component is None:
                        plot_component = "real"
                        if self._complex_funcs:
                            logger.warning("plotting only real component")
                    match plot_component:
                        case "imag":
                            plot = plt.imshow(values[0].imag, **kwargs)
                        case "real":
                            plot = plt.imshow(values[0].real, **kwargs)
                        case _:
                            raise NotImplementedError(
                                "Can't plot both imaginary and real in 2D"
                            )

            case _:
                raise NotImplementedError(
                    "plots for dimensions > 2 need to be implemented"
                )

        return plot

    @staticmethod
    def grid(res: ResType = 200) -> torch.Tensor:
        """
        grid

        grid creates a rectangular grid whose dimensions and resolution is dependent on the res parameter

        Keyword Arguments:
            res {int | slice | list[slice]} -- the resolution and dimensions of the grid passed as n slices (default: {slice(0,1,200)})

        Returns:
            torch.Tensor -- {d1,...,dn,n} tensor with n+1 dimensions where the last dimension is the coordinates and therefore of size n. all other dimensions have sizes coresponding to the resolution specified by their coresponding res parameter
        """
        if isinstance(res, int):
            res = (slice(0, 1, res),)
        elif isinstance(res, slice):
            res = (res,)
        axes = [torch.linspace(r.start, r.stop, r.step) for r in res]
        meshgrid = torch.meshgrid(axes, indexing="ij")
        return torch.stack(meshgrid, dim=-1)

    def resize_modes(
        self, target_modes: int | tuple[int, ...] | Self, rescale: bool = True
    ):
        """
        resize_modes

        creates a copy of this basis with modes resized to target modes

        Arguments:
            target_modes {int | tuple[int, ...] | Basis} -- the target mode or basis mode to resize this basis to


        Returns:
            Basis -- A copy of this basis with resized coefficients
        """
        if isinstance(target_modes, int):
            target_modes = (target_modes,)
        elif isinstance(target_modes, Basis):
            target_modes = target_modes.modes
        copy = self.copy()
        copy.coeff = resize_modes(self.coeff, target_modes, rescale=rescale)
        return copy

    def perturb(
        self,
        std_ratio: float = 0.1,
        rand_func=torch.randn,
        generator: torch.Generator | None = None,
    ) -> Self:
        """
        perturb

        add noise to the function values of the basis coefficients

        Keyword Arguments:
            std_ratio {float} -- ratio of noise to standard deviation of function values (default: {0.1})
            rand_func {Callable} -- random value generator function (default: {torch.randn})
            generator {torch.Generator | None} -- generator for reproducability (default: {None})

        Returns:
            Basis -- a perturbed copy of this basis
        """
        values = self.inv_transform(self.coeff)
        perturbed_values = (
            values
            + rand_func(
                values.shape,
                generator=generator,
                dtype=self.coeff.dtype
                if self._complex_funcs
                else self.coeff.real.dtype,
            )
            * std_ratio
            * values.std()
        )
        copy = self.copy()
        copy.coeff = self.transform(perturbed_values)
        return copy

    def __getitem__(self, indices):
        copy = self.copy()
        idx = torch.arange(len(copy))[indices]
        # ensure the indexing results in list of coefficients (0th dimension is the list index)
        copy.coeff = copy.coeff[idx].reshape((-1, *copy.modes))
        return copy


BasisSubType = TypeVar("BasisSubType", bound="Basis")
