import torch
import abc
from typing_extensions import Self
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
        super().__init__()
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

    @classmethod
    @abc.abstractmethod
    def generate(
        cls,
        n: int,
        modes: int,
        generator: torch.Generator | None = None,
        random_func=torch.randn,
    ) -> Self:
        """
        generate

        generate functions using basis functions with random coefficients

        Arguments:
            n {int} -- number of random functions to generate coefficients for.

            modes {int} -- number of coefficients in a series.

        Returns:
            Basis -- n sets of functions with coefficients with the shape (n, modes)
        """

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

    @abc.abstractmethod
    def grad(self, dim: int = 1) -> Self:
        """
        grad

        grad computes the derivative along a dimension

        Keyword Arguments:
            dim {int} -- dimension to compute gradient on (default: {1})

        Returns:
            Self -- returns an instance of current basis with antiderivative coefficients
        """

    @abc.abstractmethod
    def integral(self, dim: int = 1) -> Self:
        """
        integral

        integral computes the antiderivative along a dimension

        Keyword Arguments:
            dim {int} -- dimension to compute antiderivative on (default: {1})

        Returns:
            Self -- returns an instance of current basis with antiderivative coefficients
        """
