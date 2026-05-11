import torch
from torch.testing import assert_close
from unittest.mock import Mock, patch
from pytest import raises

from empm.stacks import Volume


def test_spharm_normalize():
    # this creates 4 rings with 9, 16, 25, 16 points each
    l_max_ = torch.tensor([2, 3, 4, 3])
    point_count = int(torch.sum((l_max_ + 1)**2).item())
    a_k_Y_ = torch.randn(point_count, dtype=torch.complex64)
    weights = torch.tensor([1., 2., 3., 4.])

    sut = Volume(l_max_ = l_max_)
    sut.a_k_Y_reco_yk_ = torch.clone(a_k_Y_)

    sut._spharm_normalize(weights)

    ratio_vec = a_k_Y_ / sut.a_k_Y_reco_yk_
    unique_ratio_real = torch.unique_consecutive(ratio_vec.real.round(decimals=4))
    unique_ratio_imag = torch.unique_consecutive(ratio_vec.imag.round(decimals=5))
    assert len(unique_ratio_real) == 1
    assert len(unique_ratio_imag) == 1
    assert unique_ratio_imag[0] == 0

    # Confirm normalization of result
    expanded_weights = weights.repeat_interleave((l_max_ + 1) ** 2)
    norm = torch.linalg.vector_norm(sut.a_k_Y_reco_yk_ * torch.sqrt(expanded_weights))
    assert abs(norm - 1.) < 1e-6

    # assert that a normalized volume does not change further
    first_norm = torch.clone(sut.a_k_Y_reco_yk_)
    sut._spharm_normalize(weights)
    assert_close(sut.a_k_Y_reco_yk_, first_norm)
