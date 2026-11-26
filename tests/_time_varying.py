import torch

from SpectralSVR import FourierBasis

UNIT = (0.0, 1.0)


def time_varying_field(seed, functions, slices, modes, device):
    generator = torch.Generator().manual_seed(seed)
    slice_values = torch.randn(functions, slices, modes, generator=generator).to(device)
    coeff = FourierBasis.transform(slice_values.flatten(0, 1)).reshape(
        functions, slices, modes
    )
    return slice_values, coeff


def slice_times(slices, device, interval=UNIT):
    start, stop = interval
    if slices == 1:
        return torch.full((1,), start, dtype=torch.float64, device=device)
    return torch.linspace(start, stop, slices, dtype=torch.float64, device=device)


def grid_points(modes, device, interval=UNIT):
    start, stop = interval
    return (
        start
        + (stop - start)
        * torch.arange(modes, dtype=torch.float64, device=device)
        / modes
    )


def rows(times, space):
    mesh = torch.meshgrid(times, space, indexing="ij")
    return torch.stack([axis.flatten() for axis in mesh], dim=-1)
