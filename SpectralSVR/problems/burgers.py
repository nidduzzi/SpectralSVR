import torch
from ..basis import Basis, BasisSubType
from ..basis.sampling import samplings_to_tuple
from ..utils import (
    Number,
    RealBand,
    SemilinearSolverSignatureType,
    WavenumberCutoff,
    WholeGrid,
    etdrk4_solver,
    resolve_device,
)
from . import Problem
from dataclasses import dataclass
from typing import Callable, Literal, Type

ParamInput = Literal["random"] | Number | torch.Tensor
ForcingCallable = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SpectralForcing:
    """A callable returning ``(n, ns)`` spectral coefficients at a time."""

    at_time: ForcingCallable

    def spectrum(self, basis: Type[BasisSubType], t: torch.Tensor) -> torch.Tensor:
        return self.at_time(t)


@dataclass(frozen=True)
class SampledForcing:
    """A callable returning ``(n, ns)`` sampled values at a time."""

    at_time: ForcingCallable

    def spectrum(self, basis: Type[BasisSubType], t: torch.Tensor) -> torch.Tensor:
        return basis.transform(self.at_time(t) + 0j)


@dataclass(frozen=True)
class BandLimited:
    """A forcing field the generator may truncate to the integration grid."""

    field: Basis


DeclaredForcing = SpectralForcing | SampledForcing
NumericalForcing = Number | DeclaredForcing | BandLimited
ForcingInput = Literal["random"] | NumericalForcing


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
        f: ForcingInput | BasisSubType = 0,
        nu: float = 0.01,
        space_domain: slice | None = None,
        time_domain: slice | None = None,
        solver: SemilinearSolverSignatureType | None = None,
        band: RealBand | None = None,
        time_dependent_coeff: bool = True,
        **kwargs,
    ) -> tuple[BasisSubType, BasisSubType]:
        """Generate a Burgers ``(u, f)`` pair, manufactured or integrated.

        ``u0="random"`` with ``f="random"`` manufactures a pair analytically and
        integrates nothing; any other combination integrates ``u0`` forward
        under ``f``. ``band`` belongs to the first, ``solver`` to the second,
        and each is refused on the other.

        The integrated branch evaluates its forcing at the integrator's stage
        times and accepts:

        - a :data:`Number`, constant in time and space;
        - :class:`SpectralForcing` or :class:`SampledForcing` wrapping a callable
          of the stage time, which say whether it returns ``(n, modes[1])``
          coefficients or values;
        - a basis field over ``(time, space)`` spectral in time
          (``time_dependent=False``), spanning the integration intervals on both
          axes and within the grid's spatial resolution;
        - :class:`BandLimited` wrapping such a field, truncating it to the grid
          on the caller's instruction.

        Refused: a field sampled per time slice, which the integrator cannot
        read between grid points; a bare tensor, which states values without
        saying when they hold; a bare callable, which does not say whether its
        result is coefficients or values.
        """
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

        space = (float(space_domain.start), float(space_domain.stop))
        time = (float(time_domain.start), float(time_domain.stop))
        T: float = time[1] - time[0]
        nt = int(time_domain.step)
        assert nt > 1, (
            f"time_domain needs at least two grid points to have a step, got {nt}"
        )
        dt = T / (nt - 1)
        t = basis.grid(time_domain).flatten().to(device=device)
        assert t[1].sub(t[0]).isclose(torch.tensor(dt)), (
            f"Make sure that the result of generating t is consistent with dt ({dt}) and t[1]-t[0] ({t[1] - t[0]})"
        )
        domain = (time, space)
        manufactured = (
            isinstance(u0, str)
            and u0 == "random"
            and isinstance(f, str)
            and f == "random"
        )
        self._reject_inapplicable(manufactured, solver, band)
        if manufactured:
            u_gen, f_gen = self.mms_solution(
                basis, n, modes, nu, domain, generator=generator, band=band
            )

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
                self._numerical_forcing(f),
                domain,
                t,
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
        f: "NumericalForcing | BasisSubType",
        domain: tuple[tuple[float, float], ...],
        t: torch.Tensor,
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

        The forcing is normalised to :meth:`_forcing_at`, a function of time
        that every stage calls, so a forcing varying within a step is seen at
        the stage times rather than frozen at the step's start.

        A manufactured forcing at cutoff ``K`` is alias-free where
        ``ns >= 6 K``, which ``ns = m`` already satisfies.
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

        forcing_at = Burgers._forcing_at(basis, f, n, ns, domain, device)

        def nonlinear(t: torch.Tensor, u_coeff: torch.Tensor) -> torch.Tensor:
            return Burgers.advection_spectrum(basis, u_coeff, d1, dealias) + forcing_at(
                t
            )

        sol = solver(linear, nonlinear, v0, t.to(device=device))  # (nt, n, ns)
        u_coeff = sol.movedim(0, 1).to(basis.coeff_dtype)  # (n, nt, ns)
        u_gen = basis(coeff=u_coeff, time_dependent=True, domain=domain)

        f_coeff = torch.stack([forcing_at(step) for step in t], dim=1).to(
            basis.coeff_dtype
        )
        f_gen = basis(coeff=f_coeff, time_dependent=True, domain=domain)
        return u_gen, f_gen

    @staticmethod
    def _forcing_at(
        basis: Type[BasisSubType],
        f: "NumericalForcing | BasisSubType",
        n: int,
        ns: int,
        domain: tuple[tuple[float, float], ...],
        device: torch.device,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Spectral forcing ``(n, ns)`` as a function of time.

        Accepts a number, a :class:`SpectralForcing` or :class:`SampledForcing`
        wrapping a callable of time, a field over ``(time, space)`` that is
        spectral in time, or that field wrapped in :class:`BandLimited`.
        """
        if isinstance(f, bool):
            raise ValueError(
                "a bool is not a forcing: True and False satisfy Number because "
                "bool subclasses int, so they would force with 1.0 and 0.0 "
                "without saying so; pass the number meant"
            )

        if isinstance(f, Number):
            constant_spectrum = basis.transform(
                torch.full((n, ns), float(f), device=device) + 0j
            )
            return lambda t: constant_spectrum

        if isinstance(f, BandLimited):
            return Burgers._field_forcing_at(
                basis, f.field, ns, domain, device, truncate=True
            )

        if isinstance(f, Basis):
            return Burgers._field_forcing_at(
                basis, f, ns, domain, device, truncate=False
            )

        if isinstance(f, DeclaredForcing):
            declared = f

            def declared_forcing_at(t: torch.Tensor) -> torch.Tensor:
                spectrum = declared.spectrum(basis, t)
                if spectrum.shape != (n, ns):
                    raise ValueError(
                        f"forcing callable must return ({n}, {ns}), got "
                        f"{tuple(spectrum.shape)}"
                    )
                return spectrum.to(device=device)

            return declared_forcing_at

        if isinstance(f, torch.Tensor):
            raise NotImplementedError(
                "numerical Burgers forcing as a bare tensor is not supported: pass "
                "a number, a declared callable, or a field spectral in time"
            )

        if callable(f):
            raise ValueError(
                "a bare callable does not say whether it returns spectral "
                "coefficients or sampled values; both have shape (n, ns) and mean "
                "different forcings, so wrap it in SpectralForcing or "
                "SampledForcing"
            )

        raise NotImplementedError(
            "numerical Burgers forcing must be a number, a declared callable, or "
            "a field spectral in time"
        )

    @staticmethod
    def _field_forcing_at(
        basis: Type[BasisSubType],
        field: Basis,
        ns: int,
        domain: tuple[tuple[float, float], ...],
        device: torch.device,
        truncate: bool,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Sample a ``(time, space)`` forcing field on the integration grid.

        Both axes must span the integration intervals, and the grid must be able
        to represent the field's spatial content: sampling content above
        ``ns // 2`` folds it onto lower wavenumbers. ``truncate`` band-limits the
        field first, and is reached only through :class:`BandLimited`.
        """
        time_interval, space_interval = domain[0], domain[1]
        if field.coeff.ndim != 3:
            raise ValueError(
                f"forcing field must span (time, space), so its coefficients are "
                f"(samples, time, space); got {field.coeff.ndim - 1} axes"
            )
        if field.time_dependent:
            raise ValueError(
                "numerical Burgers forcing must be evaluable at any time: pass "
                "a declared callable of time or a field spectral in time "
                "(time_dependent=False), not a per-time-slice field"
            )
        if field.domain[0] != time_interval:
            raise ValueError(
                f"forcing time domain {field.domain[0]} must equal the integration "
                f"interval {time_interval}"
            )
        if field.domain[1] != space_interval:
            raise ValueError(
                f"forcing space domain {field.domain[1]} must equal the integration "
                f"interval {space_interval}; the field is sampled on the "
                "integration grid, so another interval reads its values at "
                "points that mean something else on its own period"
            )
        time_modes, field_modes = field.coeff.shape[-2], field.coeff.shape[-1]
        if field_modes > ns:
            if not truncate:
                raise ValueError(
                    f"forcing field has {field_modes} spatial modes and the "
                    f"integration grid has {ns}, so sampling it there would fold "
                    "its highest wavenumbers onto lower ones; band-limit it "
                    "deliberately -- resize the field yourself, or pass "
                    "BandLimited(field) to have the generator truncate it"
                )
            representable = field.resize_modes((time_modes, ns))
        else:
            representable = field
        nodes = basis.default_sampling().nodes(ns, *space_interval).to(device=device)

        def forcing_field_at(t: torch.Tensor) -> torch.Tensor:
            query = torch.stack([torch.full_like(nodes, float(t)), nodes], dim=-1)
            return basis.transform(representable(query).to(device=device))

        return forcing_field_at

    @staticmethod
    def _numerical_forcing(
        f: "ForcingInput | BasisSubType",
    ) -> "NumericalForcing | BasisSubType":
        """Narrow ``generate``'s forcing to the forms the numerical branch takes.

        ``f="random"`` names the manufactured branch's forcing, which is derived
        from the field rather than supplied; reaching the integrator with it
        means the caller asked for a manufactured pair without a manufactured
        initial condition.
        """
        if isinstance(f, str):
            raise ValueError(
                "f='random' manufactures a forcing from the field and is only "
                "meaningful with u0='random'; the numerical branch integrates a "
                "forcing it is given, so pass a number, a callable of time, or a "
                "field spectral in time"
            )
        return f

    def mms_solution(
        self,
        basis: Type[BasisSubType],
        n: int,
        modes: tuple[int, ...],
        nu: float,
        domain: tuple[tuple[float, float], ...],
        generator: torch.Generator | None = None,
        band: RealBand | None = None,
    ) -> tuple[BasisSubType, BasisSubType]:
        """Method of Manufactured Solutions for 1D viscous Burgers.

        Pick an arbitrary band-limited solution ``u`` (a random Fourier field)
        and let the forcing be ``f = L(u)`` where ``L`` is the Burgers operator
        :meth:`spectral_residual` (with zero forcing). Then ``u`` exactly solves
        ``u_t + u u_x - nu u_xx = f`` by construction, so ``spectral_residual``
        of the returned pair is zero to machine precision.

        ``band`` states how wide the manufactured field is, defaulting to
        :meth:`widest_manufactured_band` and refused when it exceeds it.

        A :data:`RealBand` keeps a magnitude and its negative together, so the
        manufactured field is conjugate-symmetric as generated.

        Returned as non-time-dependent 2D (time, space) Fourier fields.
        """
        if len(modes) != 2:
            raise NotImplementedError("MMS Burgers is implemented for 1D space")
        bands = tuple(
            self.widest_manufactured_band(m) if band is None else band for m in modes
        )
        for axis_band, m in zip(bands, modes, strict=True):
            self._check_band(axis_band, m)
        band_limited = tuple(
            axis_band.mode_count(m) for axis_band, m in zip(bands, modes, strict=True)
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

        The largest ``K`` satisfying ``2 K <= modes / 3``, so the product stays
        inside the de-aliasing mask.
        """
        return WavenumberCutoff(modes // 6)

    @staticmethod
    def _reject_inapplicable(
        manufactured: bool,
        solver: SemilinearSolverSignatureType | None,
        band: RealBand | None,
    ) -> None:
        """Refuse an argument the dispatched branch cannot act on.

        The manufactured branch builds ``u`` analytically and integrates
        nothing; the numerical branch integrates a given pair and manufactures
        nothing. An argument meant for the other branch is refused rather than
        silently dropped.
        """
        if manufactured and solver is not None:
            raise ValueError(
                "the manufactured-solution branch (u0='random' with f='random') "
                "builds u analytically and integrates nothing, so a solver "
                "would have no effect"
            )
        if not manufactured and band is not None:
            raise ValueError(
                "the numerical branch integrates a given initial condition and "
                "forcing rather than manufacturing a field, so a band would "
                "have nothing to limit"
            )

    @staticmethod
    def highest_frequency(band: RealBand, modes: int) -> int:
        """Largest ``|k|`` ``band`` keeps on a grid of ``modes`` modes.

        A wavenumber, not a mode count. :class:`WholeGrid` keeps every bin, so
        on an even grid its largest is the self-paired Nyquist bin at
        ``modes // 2``.
        """
        match band:
            case WavenumberCutoff(cutoff=cutoff):
                return cutoff
            case WholeGrid():
                return modes // 2

    @classmethod
    def _check_band(cls, band: RealBand, modes: int) -> None:
        """Refuse a band the grid cannot hold or the de-aliasing mask would cut.

        Only a stated cutoff can ask for more modes than the grid has;
        :class:`WholeGrid` is bounded by the grid by definition, so it can reach
        only the second refusal.
        """
        if isinstance(band, WavenumberCutoff) and 2 * band.cutoff + 1 > modes:
            raise ValueError(
                f"a cutoff keeping |k| <= {band.cutoff} needs "
                f"{2 * band.cutoff + 1} modes, more than the {modes} this grid has"
            )
        widest = cls.widest_manufactured_band(modes)
        stated = cls.highest_frequency(band, modes)
        if stated > widest.cutoff:
            raise ValueError(
                f"a band keeping |k| <= {stated} squares to a product reaching "
                f"|k| <= {2 * stated}, past the |k| <= {modes / 3:.6g} the "
                f"de-aliasing mask keeps at {modes} modes; the widest admissible "
                f"band keeps |k| <= {widest.cutoff}, which is "
                f"{widest.mode_count(modes)} modes"
            )

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
