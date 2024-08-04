import pytest
from skripsi_program.model.SpectralSVR import SpectralSVR
from skripsi_program.basis import FourierBasis
from skripsi_program.utils import to_real_coeff, to_complex_coeff
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import random_split
from torchmetrics.functional import mean_squared_error


def test_SpectralSVR():
    generator = torch.Generator().manual_seed(42)
    # Generate functions/data
    # spectral density function

    # Compute coefficients for both
    n_coeffs = 2000
    modes = 7
    u_coeff_fourier = (
        FourierBasis.generateCoeff(n_coeffs, modes, generator=generator)
        * 1
        / (n_coeffs**0.5)
    )
    # ).to(dtype=torch.complex32)
    # derivative
    k = FourierBasis.waveNumber(modes)
    f_coeff_fourier = u_coeff_fourier * 2j * torch.pi * k.T
    # f_coeff_ls https://jsteinhardt.stat.berkeley.edu/blog/least-squares-and-fourier-analysis
    # u_coeff_ls
    # print(k)
    # print(u_coeff_fourier[0, :])
    # print(f_coeff_fourier[0, :])

    # Interpolate f & u
    step = 0.01
    t = torch.arange(0, 1, step)
    f_basis = FourierBasis(f_coeff_fourier)
    f = f_basis.evaluate(t)
    f = f.real

    s = torch.arange(-1, 1, step)
    u_basis = FourierBasis(u_coeff_fourier)
    u = u_basis.evaluate(s)
    u = u.real

    # Add noise

    # Train-test split
    df = TensorDataset(f, u, u_coeff_fourier)
    # df = torch.utils.data.dataset.TensorDataset(f_coeff_fourier, u, u_coeff_fourier)
    df_train, df_test = random_split(df, (0.8, 0.2), generator=generator)

    # Train svm
    periods = [1.0]
    model = SpectralSVR(FourierBasis(periods=periods), C=1.0, sigma=1.0)

    f_train, u_train, u_coeff_train = df_train[:]
    model.train(
        f_train.flatten(1), u_coeff_train.flatten(1), list(u_coeff_train.shape[1:])
    )

    # Test
    # model.test(df_test)
    f_test, u_test, u_coeff_test = df_test[:]
    assert len(f_test.shape) == 2, "f_test is more than 2 dimensional"
    if torch.is_complex(f_test):
        f_test = to_real_coeff(f_test)
    u_coeff_pred = model.svr.predict(f_test)
    u_coeff_pred = to_complex_coeff(u_coeff_pred)
    mse = mean_squared_error(to_real_coeff(u_coeff_pred), to_real_coeff(u_coeff_test))

    assert torch.isclose(
        torch.tensor(0.0), mse, atol=1e-2
    ), f"coefficient evaluation mse too high ({mse})"

    u_pred = model.forward(f_test, s).real
    # calculate mse
    mse = mean_squared_error(u_pred, u_test)
    assert torch.isclose(
        torch.tensor(0.0), mse, atol=1e-2
    ), f"prediction evaluation mse too high ({mse})"

    # Ablation Studies (maybe in another file)
