import torch
import abc
from utils import to_real_coeff, to_complex_coeff


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


if __name__ == "__main__":
    # TODO: implement tests
    # Generate Signal
    ## sampling rate
    sr = 100
    ## sampling interval
    ts = 1.0 / sr
    t = torch.arange(0, 1, ts)

    # function 1
    freq = 1.0
    f1 = 3 * torch.sin(2 * torch.pi * freq * t)
    freq = 4
    f1 += torch.sin(2 * torch.pi * freq * t)
    freq = 7
    f1 += 0.5 * torch.sin(2 * torch.pi * freq * t)

    # function 2
    freq = 1.0
    f2 = 2 * torch.sin(2 * torch.pi * freq * t)
    freq = 5
    f2 += torch.sin(2 * torch.pi * freq * t)
    freq = 10
    f2 += 0.3 * torch.sin(2 * torch.pi * freq * t)

    f = f1.unsqueeze(0)
    # f = torch.stack((f1, f2))
    f = f * (1 + 0j)  # cast to complex

    # Get coefficients and create basis
    coeff = FourierBasis.transform(f)
    basis = FourierBasis(coeff, 1)
    # derivative
    k = FourierBasis.waveNumber(basis.modes)
    f_coeff = coeff * 2j * torch.pi * k.T
    f_basis = FourierBasis(f_coeff, 1)

    # Odd samples
    # Generate Signal
    ## sampling rate
    sr = 9
    ## sampling interval
    ts = 1.0 / sr
    t = torch.arange(0, 1, ts)
    # function 1
    freq = 1.0
    f1 = 3 * torch.sin(2 * torch.pi * freq * t)
    freq = 4
    f1 += torch.sin(2 * torch.pi * freq * t)
    freq = 7
    f1 += 0.5 * torch.sin(2 * torch.pi * freq * t)

    f1 = f1.unsqueeze(-1) * (1 + 0j)

    coeff = FourierBasis.transform(f1)
    coeff_real = to_real_coeff(coeff)
    coeff_complex = to_complex_coeff(coeff_real)
    invertible = torch.equal(coeff_complex, coeff)
    assert invertible, f"coeff_complex with shape {coeff_complex.shape} and coeff with shape {coeff.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, coeff_real has shape {coeff_real.shape}"
    # Interpolate and and compare f2
    ## sampling rate
    sr = 150
    ## sampling interval
    ts = 1.0 / sr
    t = torch.arange(-1, 1, ts)
    freq = 1.0
    f3 = 3 * torch.sin(2 * torch.pi * freq * t)
    freq = 4
    f3 += torch.sin(2 * torch.pi * freq * t)
    freq = 7
    f3 += 0.5 * torch.sin(2 * torch.pi * freq * t)

    f3 = f3 * (1 + 0j)  # cast to complex

    f_pred = f_basis.evaluate(t)
    t.requires_grad_()
    pred = basis.evaluate(t)
    pred.backward(gradient=torch.ones(pred.shape, dtype=pred.dtype))
    t_grad = t.grad
    print(f"derivative difference: {torch.norm(f_pred.real - t_grad,2)}")
    # print(f_pred - t.grad)
    f3_pred = pred[0]
    assert (
        f3_pred.shape == f3.shape
    ), f"f3_pred has shape {f3_pred.shape} and f3 has shape {f3.shape}, both need to have the same shape"

    # Compare prediction with real function
    mse = torch.norm((f3_pred - f3), 2)
    print(
        f"interpolation test:\nmse: {mse.item()}, is_close: {torch.isclose(torch.tensor(0.0), mse, atol=1e-4)}"
    )
