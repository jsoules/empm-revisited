import torch
from inspect import getmembers, isfunction, signature, get_annotations
from unittest.mock import Mock, patch
from pytest import raises
from torch.testing import assert_close
from math import sqrt

from empm.stacks.Poses import Poses
from empm.parameters import Parameters



def test_update_translations():
    # This setup will ensure that displacements are capped at radius = 1
    # and will be transferred from intermediate level at radius = 0.2

    params = Parameters(delta_r_upb=1., delta_r_upd_threshold=0.4)

    # Need x- and y-displacements for 10 images, in "total",
    # "intermediate", and "current" flavors.
    # Cases to test:
    # - unchanged elements
    # - simple displacement (not triggering soft cap)
    # - soft cap updates
    # - hard cap updates, proportionally applied
    # - positive and negative displacements

    # first 2: accumulation (tmp + bit) doesn't pass threshold; no change
    # second 2: accumulation passes soft cap, change & update total
    # last 2: accumulation passes hard cap, expect total to be 1-normed w/ ratio

    total_x   = torch.tensor([0., -.5, 0., -.5, 0.75, -.5,], dtype=torch.float32)
    total_y   = torch.tensor([0., 0.5, 0., 0.5, 0.25, 0.5,], dtype=torch.float32)

    tmp_x     = torch.tensor([0., 0.3, .2, 0.3, .15,  -.25,], dtype=torch.float32)
    tmp_y     = torch.tensor([0., -.2, .2, -.2, .05,  0.25,], dtype=torch.float32)

    bit_x     = torch.tensor([0., -.1, .1, 0.1, .15,  -.1, ], dtype=torch.float32)
    bit_y     = torch.tensor([0., -.1, .1, 0.0, .05,  0.1, ], dtype=torch.float32)

    exp_tot_x = torch.tensor([0., -.5, .3, -.1, 3*sqrt(.1),  -sqrt(2)/2], dtype=torch.float32)
    exp_tot_y = torch.tensor([0., 0.5, .3, 0.3, sqrt(.1),  sqrt(2)/2], dtype=torch.float32)
    exp_tmp_x = torch.tensor([0., 0.2, 0., 0.0, 0.0,  0.], dtype=torch.float32)
    exp_tmp_y = torch.tensor([0., -.3, 0., 0.0, 0.0,  0.], dtype=torch.float32)

    exp_changed = torch.tensor([0, 0, 1, 1, 1, 1,], dtype=torch.int32)

    sut = Poses(
        n_imgs=6,
        image_delta_x_acc_M_=total_x,
        image_delta_y_acc_M_=total_y,
        image_delta_x_upd_M_=tmp_x,
        image_delta_y_upd_M_=tmp_y,
    )
    sut.image_delta_x_bit_M_ = bit_x
    sut.image_delta_y_bit_M_ = bit_y

    sut.update_translations(params)

    assert_close(sut.image_delta_x_bit_M_, torch.zeros_like(total_x))
    assert_close(sut.image_delta_y_bit_M_, torch.zeros_like(total_y))
    # raise ValueError(sut.image_delta_y_acc_M_)
    assert_close(sut.image_delta_x_acc_M_, exp_tot_x)
    assert_close(sut.image_delta_y_acc_M_, exp_tot_y)
    assert_close(sut.image_delta_x_upd_M_, exp_tmp_x)
    assert_close(sut.image_delta_y_upd_M_, exp_tmp_y)
    assert_close(sut.flag_image_delta_upd_M_, exp_changed)


def test_ctor_array_size_checks():
    other_arrays = torch.ones(12)
    shorter_array = torch.ones(10)

    fns = getmembers(Poses, predicate=isfunction)
    ctor = [x[1] for x in fns if x[0] == '__init__'][0]
    ctor_sig = signature(ctor)
    ctor_param_keys = list(ctor_sig.parameters.keys())
    param_keys = [x for x in ctor_param_keys if x != 'self' and x != 'n_imgs']

    for x in param_keys:
        other_param_keys = [y for y in param_keys if y != x]
        params_dict = dict((k, other_arrays) for k in other_param_keys)
        params_dict[x] = shorter_array
        with raises(ValueError, match="inconsistently-numbered"):
            _ = Poses(-1, **params_dict)
        with raises(ValueError, match="inconsistently-numbered"):
            _ = Poses(12, **params_dict)

    with raises(ValueError, match='without specifying'):
        _ = Poses()


def test_random_init():
    annots = get_annotations(Poses)
    rand_init_members = ['euler_polar_a_M_', 'euler_azimu_b_M_', 'euler_gamma_z_M_']
    zero_init_members = [x for x in annots.keys() if x not in rand_init_members]

    n_img = 6
    sut = Poses.random_init(n_img)
    for x in rand_init_members:
        member_under_test = getattr(sut, x)
        assert member_under_test.shape == torch.Size([n_img])
        assert member_under_test[0] != member_under_test[1]
        assert member_under_test[1] - member_under_test[0] != member_under_test[2] - member_under_test[1]
    for x in zero_init_members:
        member_under_test = getattr(sut, x)
        expected_value = 1 if x in ['flag_image_delta_upd_M_', 'image_I_value_M_'] else 0
        assert member_under_test.shape == torch.Size([n_img])
        assert torch.unique_consecutive(member_under_test).item() == expected_value
