import torch
from ..basis import Basis, BasisSubType
from ..utils import Number, euler_solver, SolverSignatureType
from . import Problem
from typing import Literal, Type

ParamInput = Literal["random"] | Number


# https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/burgersProject.pdf
# https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/heat_rk4.m
class Burgers(Problem):
    """
    Burger's equation problem for one dimension

    This class defines and generates the Burgers equation problem in one dimension. The functions themselves may be multidimensional but the derivative is only in the 0th mode dimension
    """

    def __init__(self) -> None:
        super().__init__()

    def generate(
        self,
        basis: Type[BasisSubType],
        n: int,
        modes: int | tuple[int],
        u0: ParamInput | BasisSubType = "random",
        f: ParamInput | BasisSubType = 0,
        nu: float = 0.01,
        space_domain=slice(0, 1),
        time_domain=slice(0, 1),
        solver: SolverSignatureType = euler_solver,
        timedependent_solution: bool = True,
        *args,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[BasisSubType, BasisSubType]:
        if isinstance(modes, int):
            modes = (modes,)
        assert n > 0, "number of samples n must be more than 0"
        for dim, m in enumerate(modes):
            assert m > 0, f"number of modes m must be more than 0 at dim {dim}"

        L = space_domain.stop - space_domain.start
        x = basis.grid(slice(space_domain.start, space_domain.stop, modes[0])).flatten(
            0, -2
        )
        T = time_domain.stop - time_domain.start
        nt = int(T / (0.01 * nu) + 2)
        dt = T / (nt - 1)
        t = torch.linspace(time_domain.start, time_domain.stop, nt)
        assert (
            dt == t[1] - t[0]
        ), f"Make sure that the result of generating t is consistent with dt ({dt}) and t[1]-t[0] ({t[1]-t[0]})"
        periods = (T, L)

        # Setup solution using initial condition
        if isinstance(u0, str):
            match u0:
                case "random":
                    # random first space coefficients
                    u0 = basis.generate(
                        n, modes, generator=generator, periods=periods, **kwargs
                    )
                case _:
                    raise RuntimeError(f"parameter value of u0 ({u0}) is invalid")
        elif isinstance(u0, Number):
            u0_const = float(u0)
            u0 = basis(basis.generate_empty(n, modes), periods=periods, **kwargs)
            if u0.coeff.is_complex():
                u0.coeff[:, ..., 0] = torch.tensor(u0_const + 0j)
            else:
                u0.coeff[:, ..., 0] = u0_const
        elif isinstance(u0, torch.Tensor):
            u0 = basis(u0).resize_modes(modes, rescale=False)
        elif isinstance(u0, Basis):
            u0 = u0
        else:
            raise RuntimeError("Invalid u0 value")
        # Setup forcing term
        if isinstance(f, str):
            match f:
                case "random":
                    fst = basis.generate(
                        n,
                        (*modes, *modes),
                        generator=generator,
                        periods=periods,
                        **kwargs,
                    )
                case _:
                    raise RuntimeError(f"parameter value of f ({f}) is invalid")
        elif isinstance(f, Number):
            f_const = float(f)
            fst = basis(basis.generate_empty(n, (1, *modes)), periods=periods, **kwargs)
            if fst.coeff.is_complex():
                fst.coeff[:, ..., 0] = torch.tensor(f_const + 0j)
            else:
                fst.coeff[:, ..., 0] = f_const
        elif isinstance(f, torch.Tensor):
            fst = basis(f).resize_modes((*modes, *modes), rescale=False)
        elif isinstance(f, Basis):
            fst = f
        else:
            raise RuntimeError("Invalid f value")

        def f_hat(t: torch.Tensor, x: torch.Tensor = x):
            x = x.tile((1, 2))
            x[:, 0] = t
            return basis.transform(fst(x).reshape((-1, modes[0])))

        def rhs_func(t: torch.Tensor, y0: torch.Tensor):
            return self.rhs(basis, nu, y0, f_hat(t))

        u_hat = solver(rhs_func, u0.coeff, t)
        # pad temporal boundary with zeros
        u_hat = u_hat.movedim(0, 1)
        if timedependent_solution:
            ust = basis(u_hat, **kwargs, time_dependent=True)
        else:
            u_hat = u_hat.reshape((n * nt, modes[0]))
            ust = basis(
                basis.transform(basis.inv_transform(u_hat).reshape((n, nt, modes[0]))),
                **kwargs,
            )  # .resize_modes((*modes,) * 2)

        results = (ust, fst)
        return results

    # addapted from
    # @MISC {3834917,
    #     TITLE = {Solving Viscous Burgers using spectral method},
    #     AUTHOR = {Gokul (https://math.stackexchange.com/users/827487/gokul)},
    #     HOWPUBLISHED = {Mathematics Stack Exchange},
    #     NOTE = {URL:https://math.stackexchange.com/q/3834917 (version: 2020-09-21)},
    #     EPRINT = {https://math.stackexchange.com/q/3834917},
    #     URL = {https://math.stackexchange.com/q/3834917}
    # }

    @staticmethod
    def rhs(
        basis: type[Basis],
        nu: float,
        u_hat: torch.Tensor,
        f_hat: torch.Tensor,
    ) -> torch.Tensor:
        u = basis(u_hat)
        dealias_modes = tuple(int(mode * 1.5) for mode in u.modes)
        # u_dealiased = u
        u_dealiased = u.resize_modes(dealias_modes, rescale=False)
        u_val = basis.inv_transform(u_dealiased.coeff)
        dealiased_u_u_x_hat = basis.transform(0.5 * u_val**2)
        u_u_x = basis(dealiased_u_u_x_hat).resize_modes(u.modes, rescale=False).grad()

        u_u_x_hat = u_u_x.coeff
        # if u_u_x_hat.abs().ge(20).any():
        #     print(u_val.std())
        u_xx_hat = u.grad().grad().coeff
        u_hat = nu * u_xx_hat + f_hat - u_u_x_hat
        return u_hat

    def spectral_residual(self, u: Basis, ut: Basis, *args, **kwargs) -> Basis:
        residual = super().spectral_residual(*args, **kwargs)
        raise NotImplementedError("spectral residual not implemented")
        return residual

    def residual(self, *args, **kwargs) -> Basis:
        residual = super().residual(*args, **kwargs)
        raise NotImplementedError("Function value residual not implemented")
        return residual
