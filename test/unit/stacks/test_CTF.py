import torch
from unittest.mock import Mock, patch
from torch.testing import assert_close
from pytest import mark

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


# def _make_testing_grid() -> PolarGrid:
#     # Note this imports a particular non-uniform grid whose construction
#     # is hard-coded in a test in the dir_empm package.
#     kpr_max = 48/torch.pi
#     k_eq_d = .5/torch.pi
#     template_k_eq_d = 0.5
#     (n_k_p_r, k_p_r_, wt_3d_k_p_r_) = get_weight_3d_1(0, kpr_max, k_eq_d, 'L')
#     (n_w_, wt_2d_kpr_, wt_2d_kpwk_, kpr_wk_, kpw_wk_, kc0_wk_, kc1_wk_) = get_weight_2d_2(
#         0, n_k_p_r, k_p_r_, kpr_max, template_k_eq_d, None, wt_3d_k_p_r_
#     )

#     return PolarGrid(
#         False,
#         n_k_p_r,
#         k_p_r_,
#         kpr_max,
#         template_k_eq_d,
#         n_w_,
#         wt_2d_kpr_,
#         wt_2d_kpwk_,
#         kpr_wk_,
#         kpw_wk_,
#         kc0_wk_,
#         kc1_wk_
#     )


# TODO mark it to check both list and tensor versions
@mark.parametrize("use_tensors", [False, True])
def test_make_ctfs_from_parameters(use_tensors: bool):
    volt_c = [300., 300.]
    defocusU = [22174.2, 21912.2]
    defocusV = [21393.0, 22462.6]
    defocusAngle = [1.60, 73.67]
    spherical_ab = [2.0, 2.0]
    amplitude = [0.1, 0.1]
    if use_tensors:
        volt_c = torch.tensor(volt_c)
        defocusU = torch.tensor(defocusU)
        defocusV = torch.tensor(defocusV)
        defocusAngle = torch.tensor(defocusAngle)
        spherical_ab = torch.tensor(spherical_ab)
        amplitude = torch.tensor(amplitude)
    n_pix_across = 256
    pixel_size_angstrom = 1.2156 # this *looks* sto be correct
    img_map = torch.tensor([1, 0]) if use_tensors else None

    grid = Mock()
    grid.n_w_sum = 15
    grid.k_c_0_wk_ = Mock()
    grid.k_c_1_wk_ = Mock()

    mock_ctf_1 = torch.arange(grid.n_w_sum, dtype=torch.float64)
    mock_ctf_2 = -3. * torch.arange(grid.n_w_sum, dtype=torch.float64)
    with patch("empm.stacks.CTF.niko_ctf") as m:
        m.side_effect = [(mock_ctf_1, 12), (mock_ctf_2, 15)]
        ctfs = CTF.make_ctfs_from_parameters(
            n_CTF = 2,
            grid = grid,
            n_pixels_across = n_pix_across,
            pixel_size_angstrom = pixel_size_angstrom,
            voltage_C_ = volt_c,
            defocusU_C_ = defocusU,
            defocusV_C_ = defocusV,
            defocusAngle_C_ = defocusAngle,
            sphericalAberration_C_ = spherical_ab,
            amplitudeContrast_C_ = amplitude,
            index_nCTF_from_nM_ = img_map
        )
        # TODO could do asserts about mocked-niko_ctf's call args
        # I don't think this is really that informative right now;
        # TODO need an integration test for this with actual numbers

    assert ctfs.n_CTF == 2
    assert_close(ctfs.CTF_k_p_wkC__, -1. * torch.stack([mock_ctf_1, mock_ctf_2]))
    if use_tensors:
        assert_close(ctfs.index_nCTF_from_nM_, img_map)
    else:
        assert_close(ctfs.index_nCTF_from_nM_, torch.arange(2))
