import torch
import abc
from ..utils.fourier import to_real_coeff, to_complex_coeff


# Basis functions
# - able to set number of modes / basis functions
# - provides access to the vector of basis function values evaluated at x


class Basis(abc.ABC):
    """
    Basis function for ndim dimensions
    """

    coeff: torch.Tensor
    ndim: int
    modes: int

    def __init__(self, coeff: torch.Tensor | None = None, ndim: int = 1) -> None:
        assert ndim > 0, f"ndim {ndim} is not allowed because it is less than 1"
        # assert (
        #     coef.shape[1] == 1
        # ), f"coef of shape {coef.shape} is not allowed, make sure it is one dimensional"
        self.ndim = ndim
        if coeff is not None:
            assert (
                coeff.shape[0] > 0
            ), f"coef of shape {coeff.shape} is not allowed, make sure it has at least one entry"
            self.coeff = coeff
            self.modes = coeff.shape[1]

    @abc.abstractmethod
    def fn(x: torch.Tensor, ndim: int, modes: int) -> torch.Tensor:
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

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        evaluate

        evaluate approximated function at x

        Arguments:
            x {torch.Tensor} -- m points to evaluate approximated function at.

        Returns:
            torch.Tensor -- {m, n} evaluations where n is the number different functions (coeff first dimension)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        assert self.coeff is not None, "coeff is none, set it first with setCoeff"

        return (
            1 / self.modes * torch.mm(self.coeff, self.fn(x, self.ndim, self.modes).T)
        )

    def setCoeff(self, coeff: torch.Tensor):
        self.coeff = coeff
        self.modes = coeff.shape[1]

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
    @staticmethod
    def fn(x: torch.Tensor, ndim: int, modes: int) -> torch.Tensor:
        assert (
            len(x.shape) > 1
        ), "x must have at least 2 dimensions, the format needs to be row of points, the first dimension of the tensor being each row and the second being dimensions of the points"
        assert (
            x.shape[1] == ndim
        ), f"x has shape {x.shape}, but ndim is {ndim}, make sure the second dimension of x is equal to ndim"
        assert (
            x.shape[0] > 0
        ), f"x has shape {x.shape}, make sure the first dimension isn't empty ie. has at least one row of samples"

        k = FourierBasis.waveNumber(modes).T
        return torch.exp(2j * torch.pi * k * x)  # TODO: add multidimensional support

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
        if not torch.is_complex(f):
            f = f * (1 + 0j)
        # TODO: implement multidimensional version by composing the 1D transforms
        mode = f.shape[1]
        n = torch.arange(mode)
        k = FourierBasis.waveNumber(mode)
        e = torch.exp(-2j * torch.pi * k * n / mode)

        X = torch.mm(f, e.T)

        return X

    @staticmethod
    def waveNumber(modes: int):
        k = torch.arange(modes).unsqueeze(-1) - (modes - 1) // 2
        return k

    @staticmethod
    def generateCoeff(
        n: int,
        modes: int,
        domain: tuple[float, float] = (-1.0, 1.0),
        generator: torch.Generator | None = None,
        complex: bool = True,
    ) -> torch.Tensor:
        if complex:
            modes = 2 * modes  # account for the imaginary part
        a, b = domain
        span = abs(b - a)
        coeff_real = span * torch.randn((n, modes), generator=generator) + a
        if complex:
            return to_complex_coeff(coeff_real)
        else:
            return coeff_real


## Chebyshev basis
# TODO: implement chebyshev basis https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
class ChebyshevBasis(Basis):
    def __init__(self, coef: torch.Tensor, ndim: int = 1) -> None:
        super().__init__(coef, ndim)
