import torch
from unittest.mock import Mock, patch
from torch.testing import assert_close

from empm.stacks.CTF import CTF
from empm.parameters import Parameters


def test_expand_single_value_isotropic_ctf():
    isotropic_ctf = torch.tensor([2, 3, 4, 5, 6])
    radial_dimension = isotropic_ctf.numel()
    inplane_dimension = 12
    total_points = radial_dimension * inplane_dimension

    iso_ctf_stack = torch.stack((isotropic_ctf, isotropic_ctf * 2, isotropic_ctf * 5))
    ctfs = CTF(iso_ctf_stack, torch.zeros(1))

    mock_grid = Mock()
    mock_grid.is_uniform = True
    mock_grid.n_k_p_r = radial_dimension
    mock_grid.n_w_max = inplane_dimension
    mock_grid.n_w_sum = total_points

    assert_close(ctfs.CTF_k_p_wkC__, iso_ctf_stack)

    ctfs.expand_single_value_isotropic_ctf(mock_grid)

    assert ctfs.CTF_k_p_wkC__.shape == (iso_ctf_stack.shape[0], mock_grid.n_w_sum)
    
    tmp = ctfs.CTF_k_p_wkC__.reshape((-1, mock_grid.n_k_p_r, mock_grid.n_w_max))
    tmp = torch.unique_consecutive(tmp, dim=2).squeeze()
    assert_close(tmp, iso_ctf_stack)


def test_expand_single_value_isotropic_ctf_pass_on_nonuniform_grid():
    isotropic_ctf = torch.tensor([2, 3, 4, 5, 6])
    iso_ctf_stack = torch.stack((isotropic_ctf, isotropic_ctf * 2, isotropic_ctf * 5))
    ctfs = CTF(iso_ctf_stack, torch.zeros(1))

    mock_grid = Mock()
    mock_grid.is_uniform = False

    assert_close(ctfs.CTF_k_p_wkC__, iso_ctf_stack)
    ctfs.expand_single_value_isotropic_ctf(mock_grid)
    assert_close(ctfs.CTF_k_p_wkC__, iso_ctf_stack)


def test_expand_single_value_isotropic_ctf_does_nothing_to_anisotropic_ctfs():
    mock_grid = Mock()
    mock_grid.is_uniform = True
    mock_grid.n_k_p_r = 3
    ctf_stack = torch.arange(60).reshape((4, 3, 5)) + 1.

    ctfs = CTF(ctf_stack, torch.zeros(1))

    assert_close(ctfs.CTF_k_p_wkC__, ctf_stack)
    ctfs.expand_single_value_isotropic_ctf(mock_grid)
    assert_close(ctfs.CTF_k_p_wkC__, ctf_stack)


def test_empirically_determine_rank():
    mock_S = torch.tensor([10, 1, 0.1, 0.01, 0.001, 0.0001])
    params = Parameters(tolerance_master=0.004)

    mock_grid = Mock()
    mock_grid.n_w_sum = 10
    n_M = 5
    ctf_tensor = torch.arange(mock_grid.n_w_sum * n_M).reshape((n_M, mock_grid.n_w_sum))

    ctfs = CTF(ctf_tensor, torch.zeros(12))

    with patch("empm.stacks.CTF.matlab_style_svd_macro") as mock_patch:
        mock_patch.return_value = (None, mock_S, None)
        r = ctfs.empirically_determine_rank(params, mock_grid, n_M)
        # the first 3 items of mock_S, normalized by mock_S[0], are greater
        # than the 4e-3 tolerance we set above
        assert r == 3
