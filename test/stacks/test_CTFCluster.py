import torch
from torch import Tensor
from torch.testing import assert_close
from unittest.mock import Mock, patch
from pytest import raises

from empm.grids import PolarGrid
from empm.stacks import force_isotropy, CTFCluster, ImageStack, Volume

PKG = "empm.stacks.CTFCluster"

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

    # remember that anisotropic ctfs are supposed to have linearized indices
    return (ctfs.reshape((n_ctfs, -1)), iso_ctfs)


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

    weighted_ctfs = ctfs * ctf_multipliers[:, None]
    weighted_ctfs = weighted_ctfs.reshape((n_ctfs, -1))
    mock_ctfs = _make_mock_ctf(weighted_ctfs)
    mock_ctfs.index_nCTF_from_nM_ = torch.arange(n_ctfs)

    expected_image_map_from_cluster = [[0, 2],
                                       [1, 3, 8],
                                       [4, 5, 6, 7,],
                                       [9]]
    expected_cluster_img_counts = [2, 3, 4, 1]

    with patch(f"{PKG}.knn_cluster_CTF_k_p_r_kC__1") as mock_clusterer:
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


def _get_golden_grid(expected_weights: Tensor) -> PolarGrid:
    grid = PolarGrid(
        is_uniform = True,
        n_k_p_r = 3,
        k_p_r_ = torch.tensor([3, 5, 7], dtype=torch.float32),
        k_p_r_max = None,
        template_k_eq_d = -1,
        n_w_ = torch.tensor([8,8,8], dtype=torch.int32),
        weight_2d_k_p_r_ = expected_weights**2,
        weight_2d_k_p_wk_ = torch.zeros(0),
        k_p_r_wk_ = torch.zeros(0),
        k_p_w_wk_ = torch.zeros(0),
        k_c_0_wk_ = torch.zeros(0),
        k_c_1_wk_ = torch.zeros(0)
    )
    return grid


def _get_golden_imgs(n_images: int, n_points: int, n_repeats: int = 2) -> ImageStack:
    base_range = torch.arange(n_points * n_images)
    real_part = (torch.remainder(base_range, 13) - 6).to(torch.complex64)
    imag_part = (torch.remainder(base_range, 17) - 8).to(torch.complex64)
    image_tensor = ((real_part + 1j*imag_part) / 19.).reshape((n_images, n_points))

    images = ImageStack(
        M_k_p_wkM__ = image_tensor.repeat((n_repeats, 1))
    )
    return images


# This is a golden test, alas
def test_determine_principal_modes_empirically():
    expected_weights = torch.tensor([0.25,0.50,0.75], dtype=torch.float32)
    n_images = 4
    n_clusters = 2
    grid = _get_golden_grid(expected_weights)
    images = _get_golden_imgs(n_images, int(torch.sum(grid.n_w_)), n_repeats=n_clusters)
    # Note this will give us 2 runs of the 4 images, with the same values.
    # This lets us test with multiple clusters

    expected_X_2e_Memp_d1_kkc___ = torch.tensor([
        +4.0089928109038837,
        -3.7295092288655907,
        +3.9269739377646369,
        -3.7295092288655907,
        +15.5917023474446150,
        -9.8236823757639957,
        +3.9269739377646369,
        -9.8236823757639957,
        +31.8133908051085292,], dtype=torch.float32
    ).reshape((3,3)).repeat((n_clusters, 1, 1))

    n_ctfs = n_images * n_clusters
    (ctfs, _) = _make_scaled_mock_ctf_tensor(grid.n_w_max, grid.n_k_p_r, n_ctfs, 0)
    mock_ctfs = _make_mock_ctf(ctfs)
    mock_ctfs.index_nCTF_from_nM_ = torch.arange(n_ctfs)

    # This will map each run of the images to each of the specified clusters.
    ctf_to_cluster_map = torch.repeat_interleave(
        torch.arange(n_clusters),
        torch.ones(n_clusters, dtype=torch.int32) * n_images
    )
    with patch(f"{PKG}.knn_cluster_CTF_k_p_r_kC__1") as mock_clusterer:
        mock_clusterer.return_value = (None, ctf_to_cluster_map)
        sut = CTFCluster(Mock(), grid, mock_ctfs)
        
        pm_X_kkc___ = sut._determine_principal_modes_empirically(grid, images)
        assert_close(pm_X_kkc___, expected_X_2e_Memp_d1_kkc___)
        assert_close(sut.pm_X_weight_rc__, expected_weights.reshape(1, -1).repeat((n_clusters, 1)))


def _get_golden_volume(n_y_sum: int, n_molecules: int = 2) -> Tensor:
    base_range = torch.arange(n_y_sum * n_molecules)
    real_part = (torch.remainder(base_range, 13) - 6).to(torch.complex64)
    imag_part = (torch.remainder(base_range, 17) - 8).to(torch.complex64)
    cmplx = ((real_part + 1j*imag_part) / 19.).reshape((n_molecules, n_y_sum))
    return cmplx


def test_determine_principal_modes_from_ansatz():
    expected_weights = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
    n_clusters = 2
    n_molecule = 1
    grid = _get_golden_grid(expected_weights)
    vol = Volume(l_max_ = torch.tensor([2,4,6]))
    vol.a_k_Y_reco_yk_ = _get_golden_volume(vol.n_y_sum, n_molecule)

    base_CTF_k_p_r_xavg_kc__ = torch.tensor([1,2,3], dtype=torch.float32) / torch.sqrt(torch.tensor(7.))
    ctfs = torch.stack((base_CTF_k_p_r_xavg_kc__,
                        base_CTF_k_p_r_xavg_kc__ * 2.,
                        torch.zeros_like(base_CTF_k_p_r_xavg_kc__)))
    mock_ctfs = _make_mock_ctf(ctfs)
    mock_ctfs.index_nCTF_from_nM_ = torch.arange(len(ctfs))
    ctf_to_cluster_map = torch.tensor([0, 1, 1])

    delta_sigma = 0.15
    # golden values from matlab code
    expected_pm_X_kkc___ = torch.tensor([
        +1.1715664289213095,
        -0.4917345101851578,
        +0.0037474319765092,
        -0.4917345101851578,
        +60.7408340504089423,
        -2.8076460707416753,
        +0.0037474319765092,
        -2.8076460707416855,
        +581.1505841190713681,
    ], dtype=torch.float32).reshape((3,3)).repeat((n_clusters, 1, 1))
    
    with patch(f"{PKG}.knn_cluster_CTF_k_p_r_kC__1") as mock_clusterer:
        mock_clusterer.return_value = (None, ctf_to_cluster_map)
        sut = CTFCluster(Mock(), grid, mock_ctfs)

        pm_X_kkc___ = sut._determine_principal_modes_from_ansatz(grid, vol, delta_sigma)
        assert_close(pm_X_kkc___, expected_pm_X_kkc___, atol=1.3e-6, rtol=2e-6)
        assert_close(sut.pm_X_weight_rc__, expected_weights.reshape(1, -1).repeat((n_clusters, 1)))


def test_determine_principal_modes():
    n_ctfs = 5
    n_angles = 7
    n_rings = 4
    n_clusters = 3
    grid = _make_mock_grid(n_angles, n_rings)
    (ctf_base, _) = _make_scaled_mock_ctf_tensor(n_angles, n_rings, n_ctfs)
    ctfs = _make_mock_ctf(ctf_base)
    ctfs.index_nCTF_from_nM_ = torch.arange(len(ctf_base))
    cluster_assignment = torch.tensor([0, 1, 2, 2, 2])

    parameters = Mock()
    parameters.tolerance_pm = 0.5

    mock_pm = Mock(return_value=torch.arange(n_clusters * n_ctfs).reshape((n_clusters, -1)))
    # as n_rings = 4, we expect to store 3 columns per cluster.
    # For each one we need a 4 x 4 matrix (of which we'll take max 3 columns),
    # and a singular-value vector which will determine how many elements we
    # record as actually using.
    c1_ux = torch.ones(16, dtype=torch.float32).reshape((4,4))
    c2_ux = c1_ux * 2.
    c3_ux = c1_ux * 3.
    c1_sx = torch.tensor([3., 1., 1., 0.])  # 1 sv should pass
    c2_sx = torch.tensor([4., 3., 1., 0.])  # 2 svs should pass
    c3_sx = torch.tensor([4., 3., 3., 1.])  # 3 svs should pass
    returns = [(c1_ux, c1_sx, None),
               (c2_ux, c2_sx, None),
               (c3_ux, c3_sx, None),
               ]
    expected_c1 = torch.ones(12, dtype=torch.float32).reshape(3, 4)
    expected_c2 = expected_c1 * 2
    expected_c3 = expected_c1 * 3

    expected_ranks = torch.tensor([1, 2, 3], dtype=torch.int32)
    expected_matrix = torch.stack([expected_c1, expected_c2, expected_c3])

    with patch(f"{PKG}.knn_cluster_CTF_k_p_r_kC__1") as mock_clusterer:
        mock_clusterer.return_value = (None, cluster_assignment)
        sut = CTFCluster(Mock(), grid, ctfs)
        sut._determine_principal_modes_empirically = mock_pm
        with patch(f"{PKG}.matlab_style_svd_macro") as mock_svd:
            mock_svd.side_effect = returns
            sut.determine_principal_modes(parameters, grid, Mock(), None, 0.0)

            assert_close(sut.pm_n_UX_rank_c_, expected_ranks)
            assert_close(sut.pm_UX_knc___, expected_matrix)
