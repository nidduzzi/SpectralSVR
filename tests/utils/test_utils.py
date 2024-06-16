import pytest

from skripsi_program.utils import to_complex_coeff, to_real_coeff
import torch


def test_real_invertible_even():
    # even number of coefficients
    a = torch.rand((10, 10))
    a_complex = to_complex_coeff(a)
    a_real = to_real_coeff(a_complex)
    invertible = torch.allclose(a, a_real)
    assert invertible, f"a with shape {a.shape} and a_real with shape {a_real.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, a_complex has shape {a_complex.shape}"


def test_real_invertible_odd():
    # Odd number of coefficients
    a = torch.rand((10, 9))
    a_complex = to_complex_coeff(a)
    a_real = to_real_coeff(a_complex)
    # get rid of the padding
    mask = a_real != 0.0
    mask = mask.sum(dim=0) != 0
    a_real = a_real[:, mask]
    invertible = torch.allclose(a, a_real)
    assert invertible, f"a with shape {a.shape} and a_real with shape {a_real.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, a_complex has shape {a_complex.shape}"


def test_complex_invertible_even():
    # even number of coefficients
    a = torch.concat((torch.rand((10, 9)), torch.zeros((10, 1))), 1)
    b = torch.rand((10, 10))
    c = torch.complex(a, b)
    c_real = to_real_coeff(c)
    c_complex = to_complex_coeff(c_real)
    invertible = torch.equal(c, c_complex)
    assert invertible, f"c with shape {c.shape} and c_complex with shape {c_complex.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, c_real has shape {c_real.shape}"


def test_complex_invertible_odd():
    # Odd number of coefficients
    a = torch.rand((10, 9))
    b = torch.rand((10, 9))
    c = torch.complex(a, b)
    c_real = to_real_coeff(c)
    c_complex = to_complex_coeff(c_real)
    invertible = torch.equal(c, c_complex)
    assert invertible, f"c with shape {c.shape} and c_complex with shape {c_complex.shape} are not equal, check if to_complex_coeff and to_real_coeff are producing correct results, c_real has shape {c_real.shape}"
    print("Test Passed")
