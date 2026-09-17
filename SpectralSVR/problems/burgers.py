import torch
from ..basis import Basis, BasisSubType
from ..basis.sampling import samplings_to_tuple
from ..utils import (
    Number,
    SemilinearSolverSignatureType,
    WavenumberCutoff,
    etdrk4_solver,
    resolve_device,
)
from . import Problem
from typing import Literal, Type

ParamInput = Literal["random"] | Number | torch.Tensor


# References:
# - Viscous Burgers background:
#   https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/burgersProject.pdf
#   https://www.math.unl.edu/~alarios2/courses/2017_spring_M934/documents/heat_rk4.m
# - Numerical solver: ETDRK4 (see SpectralSVR.utils.etdrk4_solver for full
#   references: Kassam & Trefethen 2005; Cox & Matthews 2002; the etdrk4cp,
#   rkstiff and SciML Burgers-spectral benchmarks).
# - Exact solutions for testing use the Cole-Hopf transformation: theta solves
#   the heat equation and u = -2*nu*theta_x/theta solves unforced Burgers.
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
        modes: int | tuple[int, ...],
        generator: torch.Generator | None = None,
        u0: ParamInput | BasisSubType = "random",
        f: ParamInput | BasisSubType = 0,
        nu: float = 0.01,
        space_domain: slice | None = None,
        time_domain: slice | None = None,
        solver: SemilinearSolverSignatureType | None = None,
        time_dependent_coeff: bool = True,
        **kwargs,
    ) -> tuple[BasisSubType, BasisSubType]:
        if space_domain is None:
            space_domain = slice(0, 1, 200)
        if time_domain is None:
            time_domain = slice(0, 1, 200)
        if isinstance(modes, int):
            modes = (modes, modes)
        assert n > 0, "number of samples n must be more than 0"
        for dim, m in enumerate(modes):
            assert m > 0, f"number of modes m must be more than 0 at dim {dim}"

        device = resolve_device()

        # per-axis (start, stop) intervals -- generation now honours a nonzero
        # (or negative) start on either axis, not just the lengths.
        space = (float(space_domain.start), float(space_domain.stop))
        time = (float(time_domain.start), float(time_domain.stop))
        T: float = time[1] - time[0]
        nt = int(time_domain.step)
        dt = T / (nt - 1)
        t = basis.grid(time_domain).flatten().to(device=device)
        assert t[1].sub(t[0]).isclose(torch.tensor(dt)), (
            f"Make sure that the result of generating t is consistent with dt ({dt}) and t[1]-t[0] ({t[1] - t[0]})"
        )
        domain = (time, space)
        if (
            isinstance(u0, str)
            and u0 == "random"
            and isinstance(f, str)
            and f == "random"
        ):
            # method of manufactured solutions: pick a band-limited solution u and
            # let the forcing be its exact Burgers residual (u0 just follows from u)
            if solver is not None:
                raise ValueError(
                    "the manufactured-solution branch (u0='random' with f='random') "
                    "builds u analytically and integrates nothing, so a solver "
                    "would have no effect"
                )
            u_gen, f_gen = self.mms_solution(
                basis, n, modes, nu, domain, generator=generator
            )

            # convert the (time, space) fields to time-dependent coefficients
            if time_dependent_coeff:
                spatial_modes = modes[1:]
                res_modes = tuple(
                    slice(space[0], space[1], mode) for mode in spatial_modes
                )

                def to_time_dependent(field: BasisSubType) -> BasisSubType:
                    val = field.get_values(res=(time_domain, *res_modes))
                    coeff = basis.transform(val.flatten(0, 1)).reshape(
                        (n, nt, *spatial_modes)
                    )
                    return basis(coeff=coeff, time_dependent=True, domain=domain)

                u_gen = to_time_dependent(u_gen)
                f_gen = to_time_dependent(f_gen)

        else:
            if not time_dependent_coeff:
                raise NotImplementedError(
                    "numerical Burgers only supports time_dependent_coeff=True"
                )
            u_gen, f_gen = self._solve_spectral(
                basis,
                n,
                modes,
                nu,
                u0,
                f,
                domain,
                t,
                nt,
                device,
                generator,
                etdrk4_solver if solver is None else solver,
            )

        results = (u_gen, f_gen)
        return results

    @staticmethod
    def advection_spectrum(
        basis: Type[BasisSubType] | type[Basis],
        u_hat: torch.Tensor,
        d1: torch.Tensor,
        dealias: torch.Tensor | float,
    ) -> torch.Tensor:
        """Spectral ``-0.5 (u^2)_x`` for the Burgers advection term.

        The single definition of the nonlinear term: the sampled field's real
        part is squared, differentiated once spectrally, and de-aliased. Every
        caller uses this so none can drift from another -- the manufactured
        forcing must be the residual of the equation the solver integrates, and
        it is only that if both form this term identically.

        Arguments:
            basis {Type[Basis]} -- basis supplying the transforms.
            u_hat {torch.Tensor} -- (samples, *modes) field in spectral space.
            d1 {torch.Tensor} -- first-derivative eigenvalues for the mode grid.
            dealias {torch.Tensor | float} -- de-aliasing mask, or 1.0 for none.

        Returns:
            torch.Tensor -- the term's spectrum, shaped like ``u_hat``.
        """
        u = basis.inv_transform(u_hat).real.to(u_hat.dtype)
        return -0.5 * d1 * basis.transform(u * u) * dealias

    @staticmethod
    def _solve_spectral(
        basis: Type[BasisSubType],
        n: int,
        modes: tuple[int, ...],
        nu: float,
        u0: "ParamInput | BasisSubType",
        f: "ParamInput | BasisSubType",
        domain: tuple[tuple[float, float], ...],
        t: torch.Tensor,
        nt: int,
        device: torch.device,
        generator: torch.Generator | None,
        solver: SemilinearSolverSignatureType,
    ) -> tuple[BasisSubType, BasisSubType]:
        """Integrate 1D viscous Burgers with ``solver`` and package the result.

        Basis-agnostic: the diagonal linear operator ``nu * d2/dx2`` and the
        spectral derivative for the nonlinear term ``0.5 (u^2)_x`` are built from
        the basis's ``derivative_eigenvalues``; the transforms use the generic
        ``transform``/``inv_transform``. Requires a basis whose differentiation
        is diagonal so that ``nu * d2`` is a plain eigenvalue array (Fourier is
        the tested case), and a uniform time grid, which ``generate`` builds
        from ``time_domain``. Solves
        ``u_t + 0.5 (u^2)_x = nu u_xx + forcing`` on the spatial domain
        ``domain[1]`` (length ``L``) with ``ns = modes[1]`` spatial modes.

        ``solver`` receives the linear operator and nonlinear term separately;
        wrap a monolithic ``rhs(t, y)`` solver in
        :func:`utils.as_semilinear_solver` to use one here.
        """
        if len(modes) != 2:
            raise NotImplementedError("numerical Burgers is implemented for 1D space")
        ns = modes[1]
        length = domain[1][1] - domain[1][0]

        d1 = basis.derivative_eigenvalues(ns, length, ord=1)
        d2 = basis.derivative_eigenvalues(ns, length, ord=2)
        if d1 is None or d2 is None:
            raise NotImplementedError(
                f"{basis.__name__} differentiation is not diagonal; the "
                "semilinear split requires a diagonally-differentiable basis "
                "(e.g. Fourier)"
            )
        d1 = d1.to(device=device)
        linear = (nu * d2).to(device=device)  # nu * u_xx eigenvalues
        mask = basis.dealias_mask(ns)
        dealias = mask.to(device=device) if mask is not None else 1.0

        # initial condition: random smooth field, a constant, or explicit values
        if isinstance(u0, str) and u0 == "random":
            v0 = basis.generate_coeff(n, ns, generator=generator).to(device=device)
        elif isinstance(u0, Number):
            field = torch.full((n, ns), float(u0), device=device) + 0j
            v0 = basis.transform(field)
        elif isinstance(u0, torch.Tensor):
            field = u0.to(device=device)
            if field.ndim == 1:
                field = field.unsqueeze(0).expand(n, ns)
            assert field.shape == (n, ns), (
                f"u0 values must be ({n}, {ns}) or ({ns},), got {tuple(field.shape)}"
            )
            v0 = basis.transform((field + 0j).clone())
        else:
            raise NotImplementedError(
                "numerical Burgers supports u0='random', a constant, or a value tensor"
            )

        # constant forcing spectrum (0 by default)
        if isinstance(f, Number):
            f_field = torch.full((n, ns), float(f), device=device) + 0j
            f_hat = basis.transform(f_field)
        else:
            raise NotImplementedError("numerical Burgers supports a constant forcing f")

        def nonlinear(ti: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return Burgers.advection_spectrum(basis, v, d1, dealias) + f_hat

        sol = solver(linear, nonlinear, v0, t.to(device=device))  # (nt, n, ns)
        u_coeff = sol.movedim(0, 1).to(basis.coeff_dtype)  # (n, nt, ns)
        u_gen = basis(coeff=u_coeff, time_dependent=True, domain=domain)

        f_coeff = f_hat.unsqueeze(1).expand(n, nt, ns).to(basis.coeff_dtype).clone()
        f_gen = basis(coeff=f_coeff, time_dependent=True, domain=domain)
        return u_gen, f_gen

    def mms_solution(
        self,
        basis: Type[BasisSubType],
        n: int,
        modes: tuple[int, ...],
        nu: float,
        domain: tuple[tuple[float, float], ...],
        generator: torch.Generator | None = None,
    ) -> tuple[BasisSubType, BasisSubType]:
        """Method of Manufactured Solutions for 1D viscous Burgers.

        Pick an arbitrary band-limited solution ``u`` (a random Fourier field)
        and let the forcing be ``f = L(u)`` where ``L`` is the Burgers operator
        :meth:`spectral_residual` (with zero forcing). Then ``u`` exactly solves
        ``u_t + u u_x - nu u_xx = f`` by construction, so ``spectral_residual``
        of the returned pair is zero to machine precision.

        ``u`` is band-limited through :meth:`widest_manufactured_band`, the
        widest cutoff whose nonlinear product survives the de-aliasing band. Any
        wider and the mask discards content the product genuinely has, and the
        pair then solves the de-aliased equation but not the continuous one.

        Returned as non-time-dependent 2D (time, space) Fourier fields.
        """
        if len(modes) != 2:
            raise NotImplementedError("MMS Burgers is implemented for 1D space")
        band = tuple(self.widest_manufactured_band(m) for m in modes)
        band_limited = tuple(
            cutoff.mode_count(m) for cutoff, m in zip(band, modes, strict=True)
        )
        u = basis.generate(
            n, band_limited, domain=domain, generator=generator
        ).resize_modes(modes, rescale=False)
        u = basis(coeff=u.coeff, domain=domain)
        zero_forcing = basis(basis.generate_empty(n, modes), domain=domain)
        f = self.spectral_residual(u, zero_forcing, nu)
        return u, f

    @staticmethod
    def widest_manufactured_band(modes: int) -> WavenumberCutoff:
        """Widest cutoff a manufactured field may use at ``modes`` modes.

        Multiplication doubles the highest frequency, so a cutoff ``K`` produces
        a product reaching ``2 K`` while the Orszag 2/3 mask keeps ``modes / 3``.
        The widest admissible cutoff is therefore the largest ``K`` satisfying
        ``2 K <= modes / 3``.
        """
        return WavenumberCutoff(modes // 6)

    def spectral_residual(
        self, u: BasisSubType, f: BasisSubType, nu: float
    ) -> BasisSubType:
        """Spectral residual of ``u_t + u u_x - nu u_xx - f``.

        ``u`` and ``f`` are a non-time-dependent Fourier field whose first mode
        axis is time and second is space. Everything is expressed through the
        generic :class:`Basis` interface (``grad`` for exact spectral
        derivatives, and ``transform``/``inv_transform`` which round-trip), so
        this works for any spectral basis, not just Fourier. For a band-limited
        manufactured solution the residual is zero to machine precision.
        """
        u_t = u.grad(dim=0, ord=1)
        u_xx = u.grad(dim=1, ord=2)

        spatial_modes = u.modes[-1]
        d1 = u.derivative_eigenvalues(spatial_modes, u.lengths[-1], ord=1)
        if d1 is None:
            raise NotImplementedError(
                f"{type(u).__name__} differentiation is not diagonal; the "
                "Burgers residual requires a diagonally-differentiable basis"
            )
        mask = u.dealias_mask(spatial_modes)
        dealias = mask.to(device=u.coeff.device) if mask is not None else 1.0
        uu_x = u.copy()
        uu_x.coeff = -self.advection_spectrum(
            type(u), u.coeff, d1.to(device=u.coeff.device), dealias
        )

        nu_u_xx = u_xx.copy()
        nu_u_xx.coeff = nu_u_xx.coeff * nu

        residual = u_t + uu_x - nu_u_xx - f
        return residual

    def residual(
        self, u: BasisSubType, f: BasisSubType, nu: float, res: int = 1000
    ) -> BasisSubType:
        u_val, grid = u.get_values_and_grid(res=res)
        f_val = f.get_values(res=res)
        dt = grid[1, 0, 0] - grid[0, 0, 0]
        dx = grid[0, 1, 1] - grid[0, 0, 1]

        # Periodicity per axis comes from the scheme get_values sampled on -- not
        # assumed. ``u.ndim`` (and thus the resolved scheme count) is the SPATIAL
        # rank: a time-dependent field strips its time axis (coeff.ndim - 2), so
        # the resolved schemes describe spatial axes only (time is a sampled
        # evolution with no Fourier scheme). A plain field keeps every axis.
        axis_schemes = samplings_to_tuple(u.sampling, u.ndim, u.default_sampling())
        if u.time_dependent:
            # values are (n, time, space): axis_schemes[0] is the SPATIAL axis
            # (not time); time is the non-periodic evolution axis.
            time_periodic, space_periodic = False, axis_schemes[0].is_periodic
        else:
            # values are (n, axis0, axis1); residual reads axis0 as time, axis1
            # as space -- both are basis axes with their own scheme.
            time_periodic = axis_schemes[0].is_periodic
            space_periodic = axis_schemes[1].is_periodic
        u_t = self.axis_diff(u_val, dt.item(), dim=1, order=1, periodic=time_periodic)
        u_xx = self.axis_diff(u_val, dx.item(), dim=2, order=2, periodic=space_periodic)
        uu_x = self.axis_diff(
            u_val.pow(2).mul(0.5), dx.item(), dim=2, order=1, periodic=space_periodic
        )

        residual_val = u_t + uu_x - nu * u_xx - f_val
        residual = u.copy()
        if u.time_dependent:
            residual.coeff = u.transform(residual_val.flatten(0, 1)).unflatten(
                0, residual_val.shape[0:2]
            )
        else:
            residual.coeff = u.transform(residual_val)

        return residual
