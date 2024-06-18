import torch


def to_complex_coeff(coeff: torch.Tensor) -> torch.Tensor:
    """
    to_complex_coeff

    convert a tensor containing rows of real coefficient vectors into rows of complex coefficient vectors

    Arguments:
        coeff {torch.Tensor} -- m by p tensor where p is twice the number modes

    Returns:
        torch.Tensor -- m by p/2 tensor of elements in complex numbers
    """
    # assert (
    #     coeff.shape[1] % 2 == 0
    # ), f"coeff has shape {coeff.shape}, make sure the number of columns are even"
    coeff_real = coeff[:, ::2]
    coeff_imag = coeff[:, 1::2]
    assert (
        coeff_real.shape[1] >= coeff_imag.shape[1]
    ), f"coeff_real has shape {coeff_real.shape} and coeff_imag has shape {coeff_imag.shape}, the second dimension coeff_real needs to be bigger or equal to the second dimension of coeff_imag"
    if coeff_real.shape[1] > coeff_imag.shape[1]:
        coeff_imag = torch.concat(
            (coeff_imag, torch.zeros((coeff_imag.shape[0], 1), dtype=coeff_imag.dtype)),
            dim=1,
        )

    converted_coeff = torch.complex(coeff_real, coeff_imag)
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
    assert torch.is_complex(
        coeff
    ), f"coeff has dtype {coeff.dtype}, make sure coeff is a complex tensor"
    converted_coeff = torch.zeros((coeff.shape[0], coeff.shape[1] * 2))
    converted_coeff[:, ::2] = coeff.real
    converted_coeff[:, 1::2] = coeff.imag

    # mask for only coefficients that are never 0
    # mask = converted_coeff != 0.0
    # mask = mask.sum(dim=0) != 0
    # converted_coeff = converted_coeff[:, mask]
    return converted_coeff


def scale_to_standard(x: torch.Tensor):
    x_real = to_real_coeff(x) if torch.is_complex(x) else x
    m = x_real.mean(0, keepdim=True)
    s = x_real.std(0, unbiased=False, keepdim=True)
    x_scaled = x_real - m
    x_scaled = x_scaled / s
    for dim in range(x.shape[1]):
        x_scaled[:, dim].nan_to_num_(m[0, dim].item())
    x_scaled = to_complex_coeff(x_scaled) if torch.is_complex(x) else x_scaled
    return x_scaled
