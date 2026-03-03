import torch
from torch import Tensor
from torch.testing import assert_close
from unittest.mock import Mock, patch
from pytest import raises

from empm.stacks import force_isotropy, CTFCluster


def _make_mock_grid(n_angles, n_rings):
    grid = Mock()
    grid.is_uniform = True
    grid.n_k_p_r = n_rings
    grid.n_w_max = n_angles
    return grid


def _make_scaled_mock_ctf_tensor(n_angles, n_rings, n_ctfs, per_ctf_multiplier: int = 1):
    if per_ctf_multiplier < 1:
        ctf_multipliers = torch.ones(n_ctfs, dtype=torch.float32)
    else:
        ctf_multipliers = (torch.arange(n_ctfs) + 1.) * per_ctf_multiplier
    
    # each ring's inplanes will be of the form 1, 2, 3, ...
    # each subsequent ring will be those values times 1, 2, 3...
    # I suppose one could configure the latter further but... why
    per_ring_vals = torch.arange(n_angles) + 1.
    ring_multipliers = torch.arange(n_rings) + 1.
    # this will give us a tensor of n_rings x n_inplanes
    base_ctf = per_ring_vals[None, :] * ring_multipliers[:, None]
    # and this just applies the average directly, resulting in a 1d tensor
    base_ctf_isotropized = torch.mean(per_ring_vals)[None] * ring_multipliers

    ctfs = base_ctf[None, :, :] * ctf_multipliers[:, None, None]
    iso_ctfs = base_ctf_isotropized * ctf_multipliers[:, None]

    return (ctfs, iso_ctfs)


def _make_mock_ctf(payload: Tensor):
    mock_ctf = Mock()
    mock_ctf.CTF_k_p_wkC__ = payload
    mock_ctf.n_CTF = len(payload)
    return mock_ctf


def test_force_isotropy_fails_for_nonuniform():
    n_angles = 7
    n_rings = 4
    mock_ctf = Mock()

    grid = _make_mock_grid(n_angles, n_rings)
    grid.is_uniform = False

    with raises(Exception, match="varying inplane angle"):
        _ = force_isotropy(mock_ctf, grid)


def test_force_isotropy():
    n_angles = 7
    n_rings = 4
    n_ctfs = 5
    grid = _make_mock_grid(n_angles, n_rings)
    (ctfs, isos) = _make_scaled_mock_ctf_tensor(n_angles, n_rings, n_ctfs)
    mock_ctf = _make_mock_ctf(ctfs)

    res = force_isotropy(mock_ctf, grid)

    assert res.shape == (n_ctfs, n_rings)
    assert_close(res, isos)


# NOTE: Can't really do this without getting the underlying clustering algorithm to work first
# Need to mock out the knn_cluster_CTF_k_p_r_kC__1() fn

def test_make_cluster_averages():
    n_rings = 3
    n_angles = 7
    n_ctfs = 10
    grid = _make_mock_grid(n_angles, n_rings)

    # setting multiplier to 0 so I get identical values
    (ctfs, isos) = _make_scaled_mock_ctf_tensor(n_angles, n_rings, n_ctfs, 0)
    ctf_to_cluster_map = torch.tensor([0, 1, 0, 1, 2, 2, 2, 2, 1, 3])
    ctf_multipliers =    torch.tensor([1, 2, 1, 3, 1, 2, 3, 4, 1, 9])
    assert len(ctf_to_cluster_map) == n_ctfs

    cluster_0_avg = (isos[0])
    cluster_1_avg = (isos[0] * torch.mean(torch.tensor([2., 3., 1.])))
    cluster_2_avg = (isos[0] * torch.mean(torch.tensor([1., 2., 3., 4.])))
    cluster_3_avg = (isos[0] * 9.)
    cluster_avgs = torch.stack((cluster_0_avg, cluster_1_avg, cluster_2_avg, cluster_3_avg))

    weighted_ctfs = ctfs * ctf_multipliers[:, None, None]
    mock_ctfs = _make_mock_ctf(weighted_ctfs)
    mock_ctfs.index_nCTF_from_nM_ = torch.arange(n_ctfs)

    expected_image_map_from_cluster = [[0, 2],
                                       [1, 3, 8],
                                       [4, 5, 6, 7,],
                                       [9]]
    expected_cluster_img_counts = [2, 3, 4, 1]

    with patch("empm.stacks.CTFCluster.knn_cluster_CTF_k_p_r_kC__1") as mock_clusterer:
        mock_clusterer.return_value = (None, ctf_to_cluster_map)
        res = CTFCluster(Mock(), grid, mock_ctfs)

        assert_close(res.index_ncluster_from_nCTF_, ctf_to_cluster_map)
        assert res.n_cluster == 4
        # this should work b/c each image maps to the same-indexed CTF in this example
        assert_close(res.index_ncluster_from_nM_, ctf_to_cluster_map)

        for i in range(len(expected_image_map_from_cluster)):
            for j in range(len(expected_image_map_from_cluster[i])):
                assert expected_image_map_from_cluster[i][j] == res.index_nM_from_ncluster__[i][j]
        for i in range(len(expected_cluster_img_counts)):
            assert expected_cluster_img_counts[i] == res.n_index_nM_from_ncluster_[i]

        assert_close(res.CTF_k_p_r_xavg_kc__, cluster_avgs)


## TODO: Move tests to a util tests directory
# (once we actually have something more useful to check)
# This suggests that the relationship I saw was spurious.
# Perhaps because the SVD is not deterministic enough for the
# transposed and non-transposed versions to line up for larger matrices?

# @mark.parametrize("mat", [
#     torch.tensor([[3., 2., 2.,], [2., 3., -2.]]),
#     torch.tensor([[3., 2., 2., 4.,], [2., 4., 3., -2.]]),
# ])
# def test_svd_macro(mat: Tensor):
#     (nat_U, nat_S, nat_V) = torch.linalg.svd(mat, full_matrices=False)
#     (res_U, res_S, res_V) = _svd_macro(mat)

#     assert_close(res_S, nat_S)
#     assert_close(nat_U * -1., res_V.T)
#     assert_close(nat_V * -1., res_U)


# def test_svd_macro_2():
#     loops = 200
#     n_svd = 20

#     for i in range(loops):
#         mat = torch.rand((55, 65)) * (i + 1)
#         print(f"{i}\n{mat}")
#         (nat_U, nat_S, nat_V) = torch.linalg.svd(mat, full_matrices=False)
#         nat_U = nat_U[:, 0:n_svd]
#         nat_S = nat_S[0:n_svd]
#         nat_V = nat_V[0:n_svd, :]
#         (res_U, res_S, res_V) = _svd_macro(mat, n_svd)

#         assert_close(res_S, nat_S)
#         assert_close(nat_U * -1., res_V.T)
#         assert_close(nat_V * -1., res_U)
