import torch
from torch.testing import assert_close
from unittest.mock import Mock
from pytest import mark
from math import sqrt, sin, cos

from empm.stacks import ImageStack, Poses
from empm.grids import PolarGrid
from dir_empm.get_weight_3d_1 import get_weight_3d_1


def test_apply_displacements_from_poses():
    grid = Mock()
    # Assume grid is 9 points with these cos and sin values:
    grid.k_c_0_wk_ = torch.tensor([1., 0.7, 0.5, 0.3, 0., -.3, -.5, -.7, -1.])
    grid.k_c_1_wk_ = torch.tensor([0., 0.3, 0.5, 0.7, 1., 0.7, 0.5, 0.3,  0.])

    # Image stack has 5 images. We displace only the 0th, 2nd, and 4th.
    poses = Poses(n_imgs=5)
    poses.image_delta_x_acc_M_ = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    poses.image_delta_y_acc_M_ = torch.tensor([-.5, -.4, -.3, -.2, -.1])
    poses.flag_image_delta_upd_M_ = torch.tensor([1, 0, 1, 0, 1])

    # Image stack then needs to be 5 x 9, 45 elements total,
    # values don't really matter but we keep them small
    image_stack = (torch.arange(45, dtype=torch.float32) / 50. + .1).reshape((5, 9))
    image_stack = image_stack.to(dtype=torch.complex64)

    # Ungainly: this is just repeating the steps of the implementation.
    # Need to find a more elegant way to test this...
    _tmp = (grid.k_c_0_wk_ * poses.image_delta_x_acc_M_[:, None] + \
            grid.k_c_1_wk_ * poses.image_delta_y_acc_M_[:, None]) * \
           poses.flag_image_delta_upd_M_[:, None]
    expected = torch.mul(torch.exp(-2j * torch.pi * _tmp), image_stack)

    # Cloning just to be sure we don't alter our source data during the test
    # (This shouldn't matter as the implementation of applying displacement
    # should *not* do it in-place but rather return a copy of the post-update array)
    sut = ImageStack(M_k_p_wkM__ = torch.clone(image_stack))
    res = sut.apply_displacements_from_poses(grid, poses)


    # confirms the mask worked--indices 1 and 3 should be unchanged
    assert_close(res[1], image_stack[1])
    assert_close(res[3], image_stack[3])
    # confirm we did anything at all
    assert image_stack[0][1] != res[0][1]

    # TODO: Is there a useful assertion I can make other than
    # this 'golden teest'... sigh
    assert_close(res[0], expected[0])
    assert_close(res[2], expected[2])
    assert_close(res[4], expected[4])


def _make_ref_grid() -> PolarGrid:
    # polar grids
    k_p_r_max = 24./torch.pi
    k_eq_d = 0.25/torch.pi
    n_w_max = 98
    (n_k_p_r, k_p_r_, weight_3d, _) = get_weight_3d_1(0, k_p_r_max, k_eq_d, 'L')
    n_w_0in_ = n_w_max * torch.ones(n_k_p_r, dtype=torch.int32)

    # NOTE: Per the original code, we are using -1 for k_eq_d instead of the value actually used
    # to create the 3d weights, b/c using the real value would result in an adaptive grid shape
    grid = PolarGrid.make_uniform_grid(n_k_p_r, k_p_r_, k_p_r_max, -1, n_w_0in_, weight_3d)
    return grid


def _make_reference_gaussian_image_stack(grid: PolarGrid, n_points_cart: int = 128, cart_diam: float = 2.0, centered: bool = True):
    pi = torch.pi
    root_2pi = sqrt(2 * pi)
    cartesian_radius = cart_diam / 2.

    # Cartesian grids
    if centered:
        grid_points = torch.linspace(-cartesian_radius, cartesian_radius, n_points_cart, dtype=torch.float32)
    else:
        grid_points = torch.linspace(-cartesian_radius, cartesian_radius, n_points_cart + 1, dtype=torch.float32)[:-1]
    inter_point_dist = grid_points[1] - grid_points[0]

    x_1__, x_0__ = torch.meshgrid(grid_points,grid_points,indexing='ij')

    # Parameters for Gaussian image
    _sigma_x = 0.0625
    _delta_ = [
        [0.75 * (0.1), 0.75 * (-0.2)],
        [0.40 * (0.2), 0.40 * (-0.05)],
        [0.50 * (-.3), 0.50 * (0.2)]
    ]

    # Create physical-cartesian images
    phys_tmp = []
    for row in _delta_:
        phys_tmp.append(torch.exp( 
            -( (x_0__ - row[0])**2 + (x_1__ - row[1])**2 ) \
           / (2 * _sigma_x ** 2)
        ) / (root_2pi * _sigma_x)**2
    )
    phys_image_stack = torch.stack(phys_tmp)

    # Manually convert physical-cartesian image stack to fourier representation
    tmp_img_fourier_polar = []
    for row in _delta_:
        for nk_p_r in range(grid.n_k_p_r):
            k_p_r = grid.k_p_r_[nk_p_r].item()
            n_w = int(grid.n_w_[nk_p_r].item())
            for nw in range(n_w):
                k_x_0 = k_p_r * cos(2 * pi * nw/n_w)
                k_x_1 = k_p_r * sin(2 * pi * nw/n_w)
                real_part = -1 * _sigma_x ** 2 * (
                    (2 * pi * k_x_0) ** 2 \
                    + (2 * pi * k_x_1) ** 2
                ) / 2
                imag_part = -2j * pi * (k_x_0 * row[0] + k_x_1 * row[1])
                point = (torch.exp(torch.tensor(real_part)) * torch.exp(torch.tensor(imag_part))).item()
                tmp_img_fourier_polar.append(point)

    fourier_image_stack = torch.tensor(tmp_img_fourier_polar, dtype=torch.complex64)
    return (phys_image_stack, fourier_image_stack, inter_point_dist)


@mark.parametrize("centered", [True, False])
def test_from_cartesian_image_stack(centered: bool):
    # Make reference images as Gaussians, in stack
    grid = _make_ref_grid()
    n_pts_cart = 128
    cart_diam = 2.0
    (phys_stack, fourier_expected, inter_point_dist) = _make_reference_gaussian_image_stack(
        grid,
        n_pts_cart,
        cart_diam,
        centered
    )

    stack = ImageStack.from_cartesian_image_stack(
        phys_stack,
        n_pts_cart,
        n_pts_cart,
        cart_diam,
        cart_diam,
        grid,
        source_is_centered=centered
    )

    # do normalization--it's ignored in actual EMPM processing, but matters for the manual-reconstruction check
    normalized_images = stack.M_k_p_wkM__ * n_pts_cart * inter_point_dist ** 2
    assert_close(normalized_images, fourier_expected.reshape((stack.n_M, -1)))
