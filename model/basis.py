import torch
import abc


# Basis functions
# - able to set number of modes / basis functions
# - provides access to the vector of basis function values evaluated at x


class Basis(abc.ABC):
    """
    Basis function for ndim dimensions
    """

    coef: torch.Tensor
    ndim: int
    mode: int

    def __init__(self, coef: torch.Tensor, ndim: int = 1) -> None:
        assert ndim > 0, f"ndim {ndim} is not allowed because it is less than 1"
        assert (
            coef.shape[0] > 0
        ), f"coef of shape {coef.shape} is not allowed, make sure it has at least one entry"
        # assert (
        #     coef.shape[1] == 1
        # ), f"coef of shape {coef.shape} is not allowed, make sure it is one dimensional"
        self.ndim = ndim
        self.coef = coef
        self.mode = coef.shape[1]

    @abc.abstractmethod
    def fn(self, mode: int, x: torch.Tensor) -> torch.Tensor:
        """
        fn

        evaluate the value of the basis functions

        Arguments:
            mode {int} -- how many basis functions from the first one to evaluate
            x {torch.Tensor} -- the m locations of ndim dimensions to evaluate the basis functions at

        Returns:
            torch.Tensor -- returns a vector using a tensor of the shape {m,n}
        """
        pass

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        evaluate

        evaluate approximated function at x

        Arguments:
            x {torch.Tensor} -- m points to evaluate approximated function at

        Returns:
            torch.Tensor -- _description_
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        return 1 / self.mode * torch.mm(self.coef, self.fn(self.mode, x).T)

    @abc.abstractmethod
    def transform(f: torch.Tensor) -> torch.Tensor:
        """
        transform

        compute basis coefficients

        Arguments:
            f {torch.Tensor} -- m vectors of descretized functions to compute the coefficients of

        Returns:
            torch.Tensor -- m vectors of coefficients
        """
        pass


## Fourier basis
class FourierBasis(Basis):
    def __init__(self, coef: torch.Tensor, ndim: int = 1) -> None:
        super().__init__(coef, ndim)

    def fn(self, mode: int, x: torch.Tensor) -> torch.Tensor:
        assert (
            len(x.shape) > 1
        ), "x must have at least 2 dimensions, the format needs to be row of points, the first dimension of the tensor being each row and the second being dimensions of the points"
        assert (
            x.shape[1] == self.ndim
        ), f"x has shape {x.shape}, but ndim is {self.ndim}"
        assert (
            x.shape[0] > 0
        ), f"x has shape {x.shape}, make sure the first dimension isn't empty ie. has at least one row of samples"

        k = FourierBasis.waveNumber(mode).T
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
        # TODO: implement multidimensional version by composing the 1D transforms
        mode = f.shape[1]
        n = torch.arange(mode)
        k = FourierBasis.waveNumber(mode)
        e = torch.exp(-2j * torch.pi * k * n / mode)

        X = torch.mm(f, e.T)

        return X

    @staticmethod
    def waveNumber(N: int):
        k = torch.arange(N).unsqueeze(-1) - N // 2
        return k


## Chebyshev basis
# TODO: implement chebyshev basis
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

    f = torch.stack((f1, f2))
    f = f * (1 + 0j)  # cast to complex

    # Get coefficients and create basis
    coeff = FourierBasis.transform(f)
    basis = FourierBasis(coeff, ndim=1)

    # Interpolate and and compare f2
    ## sampling rate
    sr = 150
    ## sampling interval
    ts = 1.0 / sr
    t = torch.arange(-1, 1, ts)
    freq = 1.0
    f3 = 2 * torch.sin(2 * torch.pi * freq * t)
    freq = 5
    f3 += torch.sin(2 * torch.pi * freq * t)
    freq = 10
    f3 += 0.3 * torch.sin(2 * torch.pi * freq * t)

    f3 = f3 * (1 + 0j)  # cast to complex

    pred = basis.evaluate(t)
    f3_pred = pred[1]
    assert (
        f3_pred.shape == f3.shape
    ), f"f3_pred has shape {f3_pred.shape} and f3 has shape {f3.shape}, both need to have the same shape"

    # Compare prediction with real function
    mse = torch.norm((f3_pred - f3), 2)
    print(
        f"mse: {mse.item()}, is_close: {torch.isclose(torch.tensor(0.0), mse, atol=1e-4)}"
    )
