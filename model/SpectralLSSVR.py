import torch.utils
import torch.utils.data
import torch.utils.data.dataset
from basis import Basis, FourierBasis
from LSSVR import LSSVR
from utils import to_complex_coeff, to_real_coeff
import torch
# model from fourier/chebyshev series
# coeficients are modeled by LSSVRs that are trained on either the input function coefficients or the discretized input function itself
# coeff . basis(x)


class SpectralLSSVR:
    def __init__(self, basis: Basis, C=10.0, sigma=1.0) -> None:
        self.basis = basis
        self.model = LSSVR(C, sigma)

    def forward(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Compute F = D(f) and evaluate F(x) where D is an abritrary operator

        Arguments:
            f {torch.Tensor} -- m discretized input functions to transform using the approximated operator
            x {torch.Tensor} -- m evaluation points for the transformed input functions

        Returns:
            torch.Tensor -- _description_
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        assert (
            f.shape[0] == x.shape[0]
        ), f"f has shape {f.shape} and x has shape {x.shape}, make sure both has the same number of rows (0th dimension)"
        # compute coefficients
        if torch.is_complex(f):
            f = to_real_coeff(f)
        coeff = self.model.predict(f)
        coeff = to_complex_coeff(coeff)

        basis_values = self.basis.fn(x, x.shape[1], self.modes)
        print(coeff[0, :])
        # compute approximated function
        return (
            1 / (self.modes) * torch.sum(coeff * basis_values, dim=-1).unsqueeze(-1)
        )  # dot product

    def train(self, df: torch.utils.data.dataset.TensorDataset):
        """
        train

        fit the lssvr to predict the output function coefficients from the input function

        Arguments:
            df {torch.utils.data.dataset.TensorDataset} -- Training dataset
        """
        f, u, u_coeff = df[:]
        self.modes = u_coeff.shape[1]

        if torch.is_complex(u_coeff):
            u_coeff = to_real_coeff(u_coeff)
        if torch.is_complex(f):
            f = to_real_coeff(f)
        self.model.fit(f, u_coeff)

    def test(self, df: torch.utils.data.dataset.TensorDataset):
        f, u, u_coeff = df[:]
        if torch.is_complex(f):
            f = to_real_coeff(f)
        u_coeff_pred = self.model.predict(f)
        u_coeff_pred = to_complex_coeff(u_coeff_pred)
        print("TEST")
        print(u_coeff_pred[0, :])
        print(u_coeff[0, :])
        mse = torch.norm(u_coeff_pred - u_coeff, 2) / u_coeff.shape[0]
        print(f"test coeff mse: {mse}")


if __name__ == "__main__":
    generator = torch.Generator().manual_seed(42)
    # TODO: Generate functions/data
    # spectral density function

    # TODO: Compute coefficients for both
    n_coeffs = 2000
    modes = 7
    u_coeff_fourier = FourierBasis.generateCoeff(n_coeffs, modes, generator=generator) * 1/(n_coeffs**.5)
    # derivative
    k = FourierBasis.waveNumber(modes)
    f_coeff_fourier = u_coeff_fourier * 2j * torch.pi * k.T
    # f_coeff_ls https://jsteinhardt.stat.berkeley.edu/blog/least-squares-and-fourier-analysis
    # u_coeff_ls
    print(k)
    print(u_coeff_fourier[0,:])
    print(f_coeff_fourier[0,:])

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

    # TODO: Train-test split
    df = torch.utils.data.dataset.TensorDataset(f, u, u_coeff_fourier)
    # df = torch.utils.data.dataset.TensorDataset(f_coeff_fourier, u, u_coeff_fourier)
    df_train, df_test = torch.utils.data.random_split(
        df, (0.8, 0.2), generator=generator
    )

    # TODO: Train svm
    model = SpectralLSSVR(FourierBasis(),0.0000001,.1)

    model.train(df_train)

    # TODO: Test
    model.test(df_test)
    f_test, u_test, u_coeff_test = df_test[:]

    indecies = torch.randint(0, u_test.shape[1] - 1, (f_test.shape[0],))
    s_sampled = s[indecies, None]
    u_sampled = u_test[torch.arange(len(indecies)), indecies]
    u_pred = model.forward(f_test, s_sampled)
    # calculate mse
    mse = torch.norm(u_pred.ravel() - u_sampled.ravel(), 2) / len(u_pred.ravel())
    print(f"evaluation mse: {mse}")

    # TODO: Ablation Studies (maybe in another file)
