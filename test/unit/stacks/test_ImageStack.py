import torch
from torch.testing import assert_close
from inspect import get_annotations, getmembers
from types import MethodType
from unittest.mock import Mock, patch
from pytest import raises

from empm.stacks import ImageStack, Poses


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

