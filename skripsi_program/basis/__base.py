import torch
import abc
from typing_extensions import Self
import matplotlib.pyplot as plt
import logging
# Basis functions
# - able to set number of modes / basis functions
# - provides access to the vector of basis function values evaluated at x

logger = logging.getLogger(__name__)


class Basis(abc.ABC):
    """
    Basis function for ndim dimensions
    """

    _coeff: torch.Tensor

    def __init__(
        self, coeff: torch.Tensor | None = None, complex_funcs: bool = False
    ) -> None:
        super().__init__()
        self.coeff = coeff
        self._complex_funcs = complex_funcs

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
    def modes(self):
        if self.ndim < 1:
            return [0]
        else:
            return self.get_modes(self.coeff)

    @staticmethod
    def get_modes(coeff: torch.Tensor):
        return list(coeff.shape[1:])

    @property
    def ndim(self):
        coeff_ndim = self.coeff.ndim
        if self.coeff is None or coeff_ndim < 1:
            return 0
        return coeff_ndim - 1

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
        self, x: torch.Tensor, *args, i: int = 0, n: int = 0, **kwargs
    ) -> torch.Tensor:
        """
        __call__

        evaluate approximated function at points x

        Arguments:
            x {torch.Tensor} -- m points to evaluate approximated function at

        Keyword Arguments:
            i {int} -- function i to start evaluations at (default: {0})
            n {int} -- n functions after function i to evaluate (default: {0} all functions)

        Returns:
            torch.Tensor -- {n, m} evaluations where n is the number different functions (coeff first dimension)
        """
        pass

    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        x: torch.Tensor,
        coeff: torch.Tensor,
        *args,
        i: int = 0,
        n: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        evaluate

        evaluate approximated function of coeff at points x

        Arguments:
            x {torch.Tensor} -- m points to evaluate approximated function at
            coeff {torch.Tensor} -- coefficients of approximated functions

        Keyword Arguments:
            i {int} -- function i to start evaluations at (default: {0})
            n {int} -- n functions after function i to evaluate (default: {0} all functions)

        Returns:
            torch.Tensor -- {n, m} evaluations where n is the number different functions (coeff first dimension)
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def transform(f: torch.Tensor) -> torch.Tensor:
        """
        transform

        compute basis coefficients

        Arguments:
            f {torch.Tensor} -- m vectors of descretized functions to compute the coefficients of.

        Returns:
            torch.Tensor -- m vectors of coefficients
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def inv_transform(f: torch.Tensor) -> torch.Tensor:
        """
        inv_transform

        compute function values from dft coefficients

        Arguments:
            f {torch.Tensor} -- m vectors of coefficeints to compute the function values of.

        Returns:
            torch.Tensor -- m vectors of function values
        """
        pass

    @classmethod
    @abc.abstractmethod
    def generate(
        cls,
        n: int,
        modes: int | list[int],
        generator: torch.Generator | None = None,
        random_func=torch.randn,
    ) -> Self:
        """
        generate

        generate functions using basis functions with random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.
            modes {int | list[int]} -- number of coefficients in a series.

        Keyword Arguments:
            generator {torch.Generator | None} -- PRNG Generator for reproducability (default: {None})
            random_func {callable} -- random function that generates the coefficients (default: {torch.randn})

        Returns:
            Basis -- n sets of functions with coefficients with the shape (n, modes)
        """

    @classmethod
    @abc.abstractmethod
    def generateCoeff(cls, n: int, modes: int) -> torch.Tensor:
        """
        generateCoeff

        generate random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.

            modes {int} -- number of coefficients in a series.

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
        return self.__class__(self.coeff)

    def __sub__(self, other: Self):
        if isinstance(other, self.__class__):
            if other.coeff is None:
                return self.copy()
            elif self.coeff is None:
                other_copy = other.copy()
                other_copy.coeff = -other.coeff.clone()
                return other_copy
            else:
                result = self.copy()
                result.coeff = self.coeff - other.coeff
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
                result = self.copy()
                result.coeff = self.coeff + other.coeff
                return result
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'"
            )

    # TODO: Add plot coefficients function

    def plot(self, i=0, n=1, res: int | slice | list[slice] = 200, plt=plt, tol=1e-5):
        """
        plot

        plot draws a plot of the functions in the basis

        Keyword Arguments:
            i {int} -- function i to start plotting (default: {0})
            n {int} -- n functions after function i to evaluate (default: {1})
            res {int | slice | list[slice]} -- function discretization resolution and domain (default: {0:1:200} domain from 0 to one on all dimensions with 200 points each)
            plt {_type_} -- Axes or pyplot to get plot or imshow from (default: {pyplot})

        Returns:
            _type_ -- returns the result of the plotting function such as list[Line2D] or AxesImage
        """
        if isinstance(res, int):
            res = [slice(0, 1, res)]
        elif isinstance(res, slice):
            res = [res]

        grid = self.grid(res).flatten(0, -2)
        values = self.__call__(grid, i=i, n=n)

        match self.ndim:
            case 1:
                color = iter(plt.colormaps["gist_rainbow"](torch.rand((len(values),))))
                if self._complex_funcs:
                    for func in values:
                        c = next(color)
                        plot = plt.plot(grid.flatten(), func.flatten().real, c=c)
                        plot = plt.plot(
                            grid.flatten(), func.flatten().imag, c=c, linestyle="dashed"
                        )
                    plt.legend(
                        [
                            (f"Real ({j+1})", f"Imaginary ({j+1})")
                            for j in range(len(values))
                        ]
                    )
                else:
                    for func in values:
                        c = next(color)
                        plot = plt.plot(grid.flatten(), func.flatten().real, c=c)
                    plt.legend([(f"Real ({j+1})") for j in range(len(values))])
            case 2:
                plot = plt.imshow(values.real.reshape((res[0].step, res[1].step)))
            case _:
                raise NotImplementedError(
                    "plots for dimensions > 2 need to be implemented"
                )

        return plot

    @staticmethod
    def grid(res: int | slice | list[slice] = 200) -> torch.Tensor:
        """
        grid

        grid creates a rectangular grid whose dimensions and resolution is dependant on the res parameter

        Keyword Arguments:
            res {int | slice | list[slice]} -- the resolution and dimensions of the grid passed as n slices (default: {slice(0,1,200)})

        Returns:
            torch.Tensor -- {d1,...,dn,n} tensor with n+1 dimensions where the last dimension is the coordinates and therefore of size n. all other dimensions have sizes coresponding to the resolution specified by their coresponding res parameter
        """
        if isinstance(res, int):
            res = [slice(0, 1, res)]
        elif isinstance(res, slice):
            res = [res]
        axes = [torch.linspace(r.start, r.stop, r.step) for r in res]
        meshgrid = torch.meshgrid(axes, indexing="xy")
        return torch.stack(meshgrid, dim=-1)
