from . import Basis
import torch
from typing_extensions import Self, Literal, Callable
from functools import partial


## Fourier basis
class FourierBasis(Basis):
    def __init__(
        self,
        coeff: torch.Tensor | None = None,
        complex_funcs: bool = False,
        periods: list[float] | None = None,
    ) -> None:
        super().__init__(coeff, complex_funcs=complex_funcs)
        self.periods = periods

    @Basis.coeff.setter
    def coeff(self, coeff: torch.Tensor | None):
        Basis.coeff.__set__(self, coeff)

    def __call__(
        self,
        x: torch.Tensor,
        coeff: torch.Tensor | None = None,
        i=0,
        n=0,
        periods: list[float] | None = None,
    ) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        if coeff is None:
            coeff = self.coeff
        assert (
            coeff is not None
        ), "coeff is none, set it in the function parameters or with setCoeff"
        if periods is None:
            periods = self.periods
        modes = self.get_modes(coeff)
        assert (
            modes is not None
        ), "modes is none, set it in the function parameters, at initialization of this basis, or via class properties"
        return self.evaluate(x, coeff, periods=periods, i=i, n=n)

    @classmethod
    def evaluate(
        cls,
        x: torch.Tensor,
        coeff: torch.Tensor,
        i=0,
        n=0,
        periods: list[float] | None = None,
    ) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        if periods is None:
            periods = [1.0 for i in range(len(coeff.shape[1:]))]

        if n > 0:
            coeff = coeff[i : i + n]
        assert coeff.is_complex(), "the coefficients passed in need to be complex"

        modes = cls.get_modes(coeff)
        basis = cls.fn(x, modes, periods=periods)
        sum_coeff_x_basis = coeff.flatten(1).mm(basis.flatten(1).t())
        scaling = 1.0 / torch.prod(torch.Tensor(modes))
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
            dim_x = x[:, dim : dim + 1].div(periods[dim])
            dim_kx = torch.mm(dim_x, k)
            if transpose:
                dim_kx = dim_kx.T
            dim_kx = dim_kx.reshape(dim_basis_shape)

            kx = kx + dim_kx

        basis = torch.exp(constant * kx)

        return basis

    @staticmethod
    def waveNumber(modes: int, dtype=torch.float):
        N = (modes - 1) // 2 + 1
        n = modes // 2
        k1 = torch.arange(0, N)
        k2 = torch.arange(-n, 0)
        k = torch.concat([k1, k2], dim=0).unsqueeze(-1).to(dtype)
        return k

    @classmethod
    def generate(
        cls,
        n: int,
        modes: int,
        range: tuple[float, float] = (0.0, 1.0),
        generator: torch.Generator | None = None,
        random_func=torch.randn,
        complex_funcs: bool = False,
    ) -> Self:
        return cls(
            cls.generateCoeff(
                n,
                modes,
                range=range,
                generator=generator,
                random_func=random_func,
                complex_funcs=complex_funcs,
            ),
            complex_funcs=complex_funcs,
        )

    @classmethod
    def generateCoeff(
        cls,
        n: int,
        modes: int,
        range: tuple[float, float] = (0.0, 1.0),
        generator: torch.Generator | None = None,
        random_func: Callable[..., torch.Tensor] = torch.randn,
        complex_funcs: bool = False,
        scale: bool = True,
    ) -> torch.Tensor:
        random_func = partial(random_func, generator=generator)
        a, b = range
        span = abs(b - a)
        if complex_funcs:
            coeff = span * random_func((n, modes), dtype=torch.complex64) + a
        else:
            rem2 = modes % 2
            is_even = rem2 == 0
            reflected_modes = (modes - 2 + rem2) // 2
            reflected_coeff = (
                span * random_func((n, reflected_modes), dtype=torch.complex64) + a
            )
            constant = span * random_func((n, 1)) + a + 0j
            # constant = constant + 0j
            coeff = torch.concatenate((constant, reflected_coeff), dim=1)
            if is_even:
                center = span * random_func((n, 1), dtype=torch.complex64) + a + 0j
                coeff = torch.concatenate((coeff, center), dim=1)
            coeff = torch.concatenate((coeff, reflected_coeff.conj().flip(1)), dim=1)

        if scale:
            coeff = coeff.mul(torch.prod(torch.tensor(modes)))
        return coeff

    @staticmethod
    def _raw_transform(
        f: torch.Tensor, func: Literal["forward", "inverse"]
    ) -> torch.Tensor:
        assert torch.is_complex(
            f
        ), "f is not complex, cast it to complex first eg. f * (1+0j)"
        N = f.shape[1]
        match func:
            case "forward":
                sign = -1
            case "inverse":
                sign = 1

        n = torch.arange(N)
        # k = FourierBasis.waveNumber(N)
        # e = torch.exp(sign * 2j * torch.pi * k * n / modes)
        e = FourierBasis.fn(
            n.view(-1, 1),
            N,
            periods=N,
            constant=sign * 2j * torch.pi,
        )

        F = torch.mm(f, e.T)

        return F

    @staticmethod
    def _ndim_transform(
        f: torch.Tensor, dim: int, func: Literal["forward", "inverse"]
    ) -> torch.Tensor:
        # flatten so that each extra dimension is treated as a separate "sample"
        # move dimension to transform to the end so that it can stay intact after f is flatened
        f_transposed = f.moveaxis(dim, -1)
        # flatten so that the last dimension is intact
        f_flatened = f_transposed.flatten(0, -2)

        F_flattened = FourierBasis._raw_transform(f_flatened, func=func)
        # unflatten so that the correct shape is returned
        F_transposed = F_flattened.reshape(f_transposed.shape)
        F = F_transposed.moveaxis(-1, dim)

        return F

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
            F = FourierBasis._raw_transform(f, "forward")
        elif ndims > 2:
            # perform 1d transform over every dimension
            F = FourierBasis._ndim_transform(f, dim=1, func="forward")
            for cdim in range(2, ndims):
                F = FourierBasis._ndim_transform(F, dim=cdim, func="forward")

        return F

    @staticmethod
    def inv_transform(F: torch.Tensor, scale=True):
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
        ndims = len(F.shape)
        assert (
            ndims >= 2
        ), f"f has shape {F.shape}, It needs to have at least two dimensions with the first being m samples"
        if not torch.is_complex(F):
            F = F * (1 + 0j)
        if ndims == 2:
            f = FourierBasis._raw_transform(F, func="inverse")
        elif ndims > 2:
            # perform 1d transform over every dimension
            f = FourierBasis._ndim_transform(F, dim=1, func="inverse")
            for cdim in range(2, ndims):
                f = FourierBasis._ndim_transform(f, dim=cdim, func="inverse")

        if scale:
            f = f.div(torch.tensor(F.shape[1:]).prod())
        return f

    def grad(self, dim: int = 1) -> Self:
        k = self.waveNumber(self.coeff.shape[1])
        multiplier = 2 * torch.pi * 1j * k.T
        coeff = self.coeff.div(multiplier)
        coeff[:, 0] = torch.tensor(0 + 0j)
        return self.__class__(coeff, periods=self.periods)

    def integral(self, dim: int = 1) -> Self:
        k = self.waveNumber(self.coeff.shape[1])
        multiplier = 2 * torch.pi * 1j * k.T
        return self.__class__(self.coeff * multiplier, periods=self.periods)

    def copy(self) -> Self:
        basis_copy = super().copy()
        basis_copy.periods = self.periods
        return basis_copy
