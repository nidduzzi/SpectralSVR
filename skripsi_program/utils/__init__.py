import torch
import logging
import typing

logger = logging.getLogger(__name__)

Number = int | float


def to_complex_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_complex_coeff

    convert a tensor containing rows of real coefficient vectors into rows of complex coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p tensor where p is twice the number modes

    Returns:
        torch.Tensor -- m by p/2 tensor of elements in complex numbers
    """
    if torch.is_complex(coeff):
        logger.warning("coeff is already complex")
        return coeff
    converted_coeff = torch.view_as_complex(coeff.reshape((*coeff.shape[:-1], -1, 2)))
    return converted_coeff


def to_mag_angle(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_mag_angle

    convert a tensor containing rows of complex coefficient vectors into rows of real coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p complex tensor where p is the number modes

    Returns:
        torch.Tensor -- m by 2p tensor of elements in real numbers
    """
    if not torch.is_complex(coeff):
        logger.warning("coeff is already real")
        return coeff
    converted_coeff = torch.empty(
        (coeff.shape[0], coeff.shape[1] * 2), dtype=coeff.real.dtype
    )
    converted_coeff[:, ::2] = coeff.abs()
    converted_coeff[:, 1::2] = coeff.angle()

    return converted_coeff


def to_real_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_real_coeff

    convert a tensor containing rows of complex coefficient vectors into rows of real coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p complex tensor where p is the number modes

    Returns:
        torch.Tensor -- m by 2p tensor of elements in real numbers
    """
    if not torch.is_complex(coeff):
        logger.warning("coeff is already real")
        return coeff
    converted_coeff = torch.view_as_real(coeff).flatten(-2)

    return converted_coeff


class StandardScaler:
    def __init__(self):
        pass

    @staticmethod
    def _get_tensor_tuple(xs: tuple[torch.Tensor, ...] | torch.Tensor):
        if isinstance(xs, torch.Tensor):
            return (xs,)
        else:
            return xs

    @staticmethod
    def _get_m(x: torch.Tensor):
        x_real = to_real_coeff(x) if torch.is_complex(x) else x
        m = x_real.mean(0, keepdim=True)
        return m

    @staticmethod
    def _get_s(x: torch.Tensor, eps: float):
        x_real = to_real_coeff(x) if torch.is_complex(x) else x
        s = x_real.std(0, unbiased=False, keepdim=True)
        s[s <= eps] = eps
        return s

    def fit(self, xs: tuple[torch.Tensor, ...] | torch.Tensor, eps=1e-12):
        xs = self._get_tensor_tuple(xs)
        self.ms = tuple(self._get_m(x) for x in xs)
        self.ss = tuple(self._get_s(x, eps) for x in xs)
        self.xs_is_complex = tuple(torch.is_complex(x) for x in xs)
        self.xs_dims = tuple(x.shape[1] for x in xs)
        return self

    @staticmethod
    def _translate(x: torch.Tensor, m: torch.Tensor, i: int | None = None):
        x_name = "x" if i is None else f"xs[{i}]"
        assert (
            x.shape[1] == m.shape[1]
        ), f"{x_name} needs to have the same 2nd dimension now and at fitting"
        x_translated = x.subtract(m)
        return x_translated

    @staticmethod
    def _scale(x: torch.Tensor, s: torch.Tensor, i: int | None = None):
        x_name = "x" if i is None else f"xs[{i}]"
        assert (
            x.shape[1] == s.shape[1]
        ), f"{x_name} needs to have the same 2nd dimension now and at fitting"
        x_scaled = x.div(s)
        return x_scaled

    def _check_consistency(self, xs: tuple[torch.Tensor, ...]):
        xs_is_complex = tuple(torch.is_complex(x) for x in xs)
        assert (
            xs_is_complex == self.xs_is_complex
        ), f"make sure the complex tensors are the same order when fitting, current complex: {xs_is_complex}, fitting complex: {self.xs_is_complex}"
        xs_dims = tuple(x.shape[1] for x in xs)
        assert (
            xs_dims == self.xs_dims
        ), f"make sure the tensors have the same 2nd dimensions with the ones used at fitting, current dims: {xs_dims}, fitting dims: {self.xs_dims}"

    def get_subset_scaler(self, indices: int | list[int]):
        if isinstance(indices, int):
            indices = [indices]
        new_scaler = StandardScaler()
        new_scaler.ms = tuple(self.ms[index] for index in indices)
        new_scaler.ss = tuple(self.ss[index] for index in indices)
        new_scaler.xs_is_complex = tuple(self.xs_is_complex[index] for index in indices)
        new_scaler.xs_dims = tuple(self.xs_dims[index] for index in indices)
        return new_scaler

    @typing.overload
    def transform(self, xs: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def transform(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    def transform(self, xs: tuple[torch.Tensor, ...] | torch.Tensor):
        is_tensor_input = isinstance(xs, torch.Tensor)
        xs = self._get_tensor_tuple(xs)
        self._check_consistency(xs)
        xs_real = tuple(
            to_real_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs)
        )
        xs_transformed = tuple(
            self._translate(x, m, i=i) for i, (x, m) in enumerate(zip(xs_real, self.ms))
        )
        xs_transformed = tuple(
            self._scale(x, s, i=i)
            for i, (x, s) in enumerate(zip(xs_transformed, self.ss))
        )
        for x, m in zip(xs_transformed, self.ms):
            for dim in range(x.shape[1]):
                x[:, dim].nan_to_num_(m[0, dim].item())

        xs_out = tuple(
            to_complex_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs_transformed)
        )
        return xs_out[0] if is_tensor_input else xs_out

    @typing.overload
    def inverse(self, xs: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def inverse(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    def inverse(self, xs: tuple[torch.Tensor, ...] | torch.Tensor):
        is_tensor_input = isinstance(xs, torch.Tensor)
        xs = self._get_tensor_tuple(xs)
        self._check_consistency(xs)
        xs_real = tuple(
            to_real_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs)
        )
        xs_transformed = tuple(
            self._scale(x, 1.0 / s, i=i)
            for i, (x, s) in enumerate(zip(xs_real, self.ss))
        )
        xs_transformed = tuple(
            self._translate(x, -1.0 * m, i=i)
            for i, (x, m) in enumerate(zip(xs_transformed, self.ms))
        )
        for x, m in zip(xs_transformed, self.ms):
            for dim in range(x.shape[1]):
                x[:, dim].nan_to_num_(m[0, dim].item())
        xs_out = tuple(
            to_complex_coeff(x) if x_is_complex else x
            for x_is_complex, x in zip(self.xs_is_complex, xs_transformed)
        )
        return xs_out[0] if is_tensor_input else xs_out

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str):
        scaler: StandardScaler = torch.load(path)
        assert isinstance(
            scaler, StandardScaler
        ), "Loaded object is not a valid instance of StandardScaler"
        return scaler


def scale_to_standard(x: torch.Tensor):
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled


def resize_modes(x: torch.Tensor, target_modes: int | tuple[int, ...], rescale=True):
    if isinstance(target_modes, int):
        target_modes = (target_modes,) * len(x.shape[1:])
    assert (
        len(x.shape[1:]) == len(target_modes)
    ), f"x and max_modes should be the same dimensions after the first dimension of x, x has shape {x.shape} and max_modes is {target_modes}"
    x_resized = x
    for dim, new_mode in enumerate(target_modes, 1):
        current_mode = x.shape[dim]
        if new_mode < current_mode:
            start_range = torch.tensor(range((new_mode - 1) // 2 + 1))
            end_range = torch.tensor(range(current_mode - new_mode // 2, current_mode))

            x_resized = torch.concat(
                (
                    x_resized.index_select(dim, start_range),
                    x_resized.index_select(dim, end_range),
                ),
                dim,
            )
        elif new_mode > current_mode:
            start_range = torch.tensor(range((current_mode - 1) // 2 + 1))
            end_range = torch.tensor(range(current_mode // 2, current_mode))
            padding_size = new_mode - current_mode
            modes = list(x_resized.shape)
            modes[dim] = padding_size
            padding = torch.zeros(modes).to(x_resized)

            x_resized = torch.concat(
                (
                    x_resized.index_select(dim, start_range),
                    padding,
                    x_resized.index_select(dim, end_range),
                ),
                dim,
            )

    if rescale:
        modes = list(x.shape[1:])
        x_resized = (
            x_resized
            * torch.prod(torch.tensor(target_modes))
            / torch.prod(torch.tensor(modes))
        )

    return x_resized


def zero_coeff(x: torch.Tensor, zeroed_modes: int | list[int]):
    if isinstance(zeroed_modes, int):
        zeroed_modes = [zeroed_modes] * len(x.shape[1:])
    assert (
        len(x.shape[1:]) == len(zeroed_modes)
    ), f"x and max_modes should be the same dimensions after the first dimension of x, x has shape {x.shape} and max_modes is {zeroed_modes}"
    x_zeroed = x.clone()
    for dim, max_mode in enumerate(zeroed_modes, 1):
        dim_len = x.shape[dim]
        zero_range = torch.tensor(
            range((max_mode - 1) // 2 + 1, dim_len - max_mode // 2)
        )
        x_zeroed.index_fill_(dim, zero_range, torch.zeros((1,), dtype=x.dtype)[0])

    return x_zeroed


def mse(u_pred: torch.Tensor, u: torch.Tensor, mean=True):
    mse = (u - u_pred).pow(2).sum(0)
    if mean:
        return mse.mean()
    else:
        return mse


def rmse(u_pred: torch.Tensor, u: torch.Tensor, mean=True):
    rmse = mse(u_pred, u, mean=False).pow(0.5)
    if mean:
        return rmse.mean()
    else:
        return rmse


def r2_score(u_pred: torch.Tensor, u: torch.Tensor, mean=True):
    u_mean = u.mean(0)
    ssr = mse(u_pred, u)
    sst = mse(u, u_mean)
    ssr_sst = ssr / sst

    r2 = torch.nan_to_num(1 - ssr_sst)
    if mean:
        return r2.mean()
    else:
        return r2


def r2_expected_score(u_pred: torch.Tensor, u: torch.Tensor, mean=True):
    u_mean = u.mean(0)
    ssg = mse(u_pred, u_mean)
    sst = mse(u, u_mean)
    ssg_sst = ssg / sst

    r2_expected = torch.nan_to_num(1 - ssg_sst)
    if mean:
        return r2_expected.mean()
    else:
        return r2_expected
