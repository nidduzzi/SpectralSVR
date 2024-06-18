import torch
import abc
from ..utils import to_complex_coeff
from typing import Literal


# Basis functions
# - able to set number of modes / basis functions
# - provides access to the vector of basis function values evaluated at x


class Basis(abc.ABC):
    """
    Basis function for ndim dimensions
    """

    coeff: torch.Tensor | None
    ndim: int
    modes: list[int] | None

    def __init__(
        self,
        coeff: torch.Tensor | None = None,
    ) -> None:
        # assert ndim > 0, f"ndim {ndim} is not allowed because it is less than 1"
        # assert (
        #     coef.shape[1] == 1
        # ), f"coef of shape {coef.shape} is not allowed, make sure it is one dimensional"
        # self.ndim = ndim
        if coeff is not None:
            self.setCoeff(coeff)

    @staticmethod
    @abc.abstractmethod
    def fn(
        x: torch.Tensor,
        modes: int | list[int],
        periods: int | float | list[float] | list[int] | None = None,
        constant=None,
        transpose: bool = False,
    ) -> torch.Tensor:
        """
        fn

        evaluate the value of the basis functions

        Arguments:
            x {torch.Tensor} -- the m locations of ndim dimensions to evaluate the basis functions at.

            ndim {int} -- dimensions of the basis functions.

            modes {int} -- number of basis functions from the first one to evaluate.

        Returns:
            torch.Tensor -- returns a vector using a tensor of the shape {m,modes}
        """
        pass

    @abc.abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        evaluate

        evaluate approximated function at x

        Arguments:
            x {torch.Tensor} -- m points to evaluate approximated function at.

        Returns:
            torch.Tensor -- {m, n} evaluations where n is the number different functions (coeff first dimension)
        """
        pass

    def setCoeff(self, coeff: torch.Tensor):
        assert (
            coeff.shape[0] > 0
        ), f"make sure coeff has at least one entry, coef of shape {coeff.shape} is not allowed"
        assert coeff.shape[1] > 0, "coeff needs to be a two dimensional tensor"
        self.coeff = coeff
        self.modes = list(coeff.shape[1:])

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

    @staticmethod
    @abc.abstractmethod
    def generateCoeff(n: int, modes: int) -> torch.Tensor:
        """
        generateCoeff

        generate random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.

            modes {int} -- number of coefficients in a series.

        Returns:
            torch.Tensor -- n sets of coefficients with the shape (n, modes)
        """


## Fourier basis
class FourierBasis(Basis):
    def __init__(
        self,
        coeff: torch.Tensor | None = None,
        periods: list[float] | None = None,
    ) -> None:
        super().__init__(coeff)
        self.periods = periods

    def evaluate(
        self,
        x: torch.Tensor,
        coeff: torch.Tensor | None = None,
        periods: list[float] | None = None,
        modes: list[int] | None = None,
    ) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        if coeff is None:
            coeff = self.coeff
        assert (
            coeff is not None
        ), "coeff is none, set it in the function parameters or with setCoeff"

        if periods is None:
            periods = (
                [1.0 for i in range(len(coeff.shape[1:]))]
                if self.periods is None
                else self.periods
            )
        assert (
            periods is not None
        ), "periods is none, set it in the function parameters, at initialization of this basis, or via class properties"
        if modes is None:
            modes = self.modes
        assert (
            modes is not None
        ), "modes is none, set it in the function parameters, at initialization of this basis, or via class properties"

        coeff_x_basis = coeff.unsqueeze(1) * self.fn(
            x, modes, periods=periods
        ).unsqueeze(0)
        sum_coeff_x_basis = coeff_x_basis.flatten(2).sum(2)
        scaling = 1.0 / torch.prod(torch.Tensor(self.modes))
        return scaling * sum_coeff_x_basis

    @staticmethod
    def fn(
        x: torch.Tensor,
        modes: int | list[int],
        periods: int | float | list[float] | list[int] | None = None,
        constant=2j * torch.pi,
        transpose: bool = False,
    ) -> torch.Tensor:
        if isinstance(modes, int):
            modes = [modes]
        if isinstance(periods, int):
            periods = [periods]
        if isinstance(periods, float):
            periods = [periods]
        if periods is None:
            periods = [1.0 for i in range(len(modes))]

        assert (
            len(x.shape) > 1
        ), "x must have at least 2 dimensions, the format needs to be row of points, the first dimension of the tensor being each row and the second being dimensions of the points"
        assert (
            x.shape[0] > 0
        ), f"x has shape {x.shape}, make sure the first dimension isn't empty ie. has at least one row of samples"
        assert (
            x.shape[1] == len(modes)
        ), f"x has dimensions {x.shape[1]} and modes has dimensions {len(modes)}, both need to have the same dimensions (modes specify how many modes in each dimension of the fourier series)"
        assert (
            x.shape[1] == len(periods)
        ), f"x has dimensions {x.shape[1]} and periods has dimensions {len(periods)}, both need to have the same dimensions (periods the function periodicity in each dimension)"
        ndims = x.shape[1]

        # Compute the Fourier basis functions
        # one time for each dimension

        dim = 0
        num_modes = modes[dim]
        dim_basis_shape = [1 for i in range(ndims + 1)]
        dim_basis_shape[0] = x.shape[0]
        dim_basis_shape[dim + 1] = num_modes
        kx = torch.zeros((x.shape[0], modes[0])).reshape(dim_basis_shape)

        for dim, num_modes in enumerate(modes):
            # for dim, num_modes in list(enumerate(modes))[::-1]:
            k = FourierBasis.waveNumber(num_modes).T
            dim_basis_shape = [1 for i in range(ndims + 1)]
            dim_basis_shape[0] = x.shape[0]
            dim_basis_shape[dim + 1] = num_modes
            dim_kx = k * x[:, dim : dim + 1] / periods[dim]
            if transpose:
                dim_kx = dim_kx.T
            dim_kx = dim_kx.reshape(dim_basis_shape)

            kx = kx + dim_kx

        basis = torch.exp(constant * kx)

        return basis

    @staticmethod
    def waveNumber(modes: int):
        N = (modes - 1) // 2 + 1
        n = modes // 2
        k1 = torch.arange(0, N)
        k2 = torch.arange(-n, 0)
        k = torch.concat([k1, k2], dim=0).unsqueeze(-1)
        return k

    @staticmethod
    def generateCoeff(
        n: int,
        modes: int,
        domain: tuple[float, float] = (-1.0, 1.0),
        generator: torch.Generator | None = None,
        random_func=torch.randn,
        complex: bool = True,
    ) -> torch.Tensor:
        if complex:
            modes = 2 * modes  # account for the imaginary part
        a, b = domain
        span = abs(b - a)
        coeff_real = span * random_func((n, modes), generator=generator) + a
        if complex:
            return to_complex_coeff(coeff_real)
        else:
            return coeff_real

    @staticmethod
    def _raw_transform(
        x: torch.Tensor, func: Literal["forward", "inverse"]
    ) -> torch.Tensor:
        assert torch.is_complex(
            x
        ), "f is not complex, cast it to complex first eg. f * (1+0j)"
        modes = x.shape[1]
        match func:
            case "forward":
                sign = -1
            case "inverse":
                sign = 1

        n = torch.arange(modes)
        k = FourierBasis.waveNumber(modes)
        e = torch.exp(sign * 2j * torch.pi * k * n / modes)

        X = torch.mm(x, e.T)

        return X

    @staticmethod
    def _ndim_transform(
        f: torch.Tensor, dim: int, func: Literal["forward", "inverse"]
    ) -> torch.Tensor:
        # flatten so that each extra dimension is treated as a separate "sample"
        # move dimension to transform to the end so that it can stay intact after f is flatened
        f_transposed = f.transpose(dim, -1)
        # flatten so that the last dimension is intact
        f_flatened = f_transposed.flatten(0, -2)

        X_flattened = FourierBasis._raw_transform(f_flatened, func=func)
        # unflatten so that the correct shape is returned
        X_transposed = X_flattened.reshape(f_transposed.shape)
        X = X_transposed.transpose(-1, dim)

        return X

    @staticmethod
    def transform(f: torch.Tensor) -> torch.Tensor:
        """
        transform

        Function to calculate the
        discrete Fourier Transform
        of a real-valued signal f

        Arguments:
            f {torch.Tensor} -- m discretized real valued functions

        Returns:
            torch.Tensor -- m complex valued coefficients of f
        """
        ndims = len(f.shape)
        assert (
            ndims >= 2
        ), f"f has shape {f.shape}, It needs to have at least two dimensions with the first being m samples"
        if not torch.is_complex(f):
            f = f * (1 + 0j)
        if ndims == 2:
            X = FourierBasis._raw_transform(f, "forward")
        elif ndims > 2:
            # perform 1d transform over every dimension
            X = FourierBasis._ndim_transform(f, dim=1, func="forward")
            for cdim in range(2, ndims):
                X = FourierBasis._ndim_transform(X, dim=cdim, func="forward")

        return X

    @staticmethod
    def inv_transform(coeff: torch.Tensor):
        """
        transform

        Function to calculate the
        discrete Fourier Transform
        of a real-valued signal f

        Arguments:
            f {torch.Tensor} -- m discretized real valued functions

        Returns:
            torch.Tensor -- m complex valued coefficients of f
        """
        ndims = len(coeff.shape)
        assert (
            ndims >= 2
        ), f"f has shape {coeff.shape}, It needs to have at least two dimensions with the first being m samples"
        if not torch.is_complex(coeff):
            coeff = coeff * (1 + 0j)
        if ndims == 2:
            X = FourierBasis._raw_transform(coeff, func="inverse")
        elif ndims > 2:
            # perform 1d transform over every dimension
            X = FourierBasis._ndim_transform(coeff, dim=1, func="inverse")
            for cdim in range(2, ndims):
                X = FourierBasis._ndim_transform(X, dim=cdim, func="inverse")

        return X


## Chebyshev basis
# TODO: implement chebyshev basis https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
class ChebyshevBasis(Basis):
    def __init__(self, coef: torch.Tensor, ndim: int = 1) -> None:
        super().__init__(coef)
