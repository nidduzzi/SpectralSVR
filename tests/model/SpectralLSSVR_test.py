import pytest
from skripsi_program.model.SpectralLSSVR import SpectralLSSVR
from skripsi_program.basis import FourierBasis
from skripsi_program.utils.fourier import to_real_coeff, to_complex_coeff
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import random_split


def test_SpectralLSSVR():
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
    model = SpectralLSSVR(FourierBasis(), 1.0, 1.0)

    model.train(df_train)

    # Test
    # model.test(df_test)
    f_test, u_test, u_coeff_test = df_test[:]
    if torch.is_complex(f_test):
        f_test = to_real_coeff(f_test)
    u_coeff_pred = model.lssvr.predict(f_test)
    u_coeff_pred = to_complex_coeff(u_coeff_pred)
    mse = torch.norm(u_coeff_pred - u_coeff_test, 2) / u_coeff_test.shape[0]

    assert torch.isclose(
        torch.tensor(0.0), mse, atol=1e-2
    ), f"coefficient evaluation mse too high ({mse})"

    indecies = torch.randint(0, u_test.shape[1] - 1, (f_test.shape[0],))
    s_sampled = s[indecies, None]
    u_sampled = u_test[torch.arange(len(indecies)), indecies]
    u_pred = model.forward(f_test, s_sampled)
    # calculate mse
    mse = torch.norm(u_pred.ravel() - u_sampled.ravel(), 2) / len(u_pred.ravel())
    assert torch.isclose(
        torch.tensor(0.0), mse, atol=1e-2
    ), f"prediction evaluation mse too high ({mse})"

    # Ablation Studies (maybe in another file)
