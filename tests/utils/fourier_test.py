import pytest

from skripsi_program.utils.fourier import to_complex_coeff, to_real_coeff
import torch


def test_even():
    # even number of coefficients
    a = torch.randn((10, 10))
    a_complex = to_complex_coeff(a)
    a_real = to_real_coeff(a_complex)
    invertible = torch.equal(a, a_real)
    assert invertible, f"a with shape {a.shape} and a_real with shape {a_real.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, a_complex has shape {a_complex.shape}"
    
def test_odd():
    # Odd number of coefficients
    a = torch.randn((10, 9))
    a_complex = to_complex_coeff(a)
    a_real = to_real_coeff(a_complex)
    invertible = torch.equal(a, a_real)
    assert invertible, f"a with shape {a.shape} and a_real with shape {a_real.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, a_complex has shape {a_complex.shape}"