import torch
from unittest.mock import Mock, patch
from torch.testing import assert_close
from pytest import mark
from numpy import float64, ceil

from empm.stacks import CartesianVolume, Volume
from empm.parameters import Parameters

from dir_empm.plane_wave_expansion_1 import plane_wave_expansion_1
from dir_empm.sample_sphere_7 import sample_sphere_7
from dir_empm.convert_spharm_to_x_c_uniform_over_n_k_p_r_5 import convert_spharm_to_x_c_uniform_over_n_k_p_r_5

def _planewave_integral(displacement_vector: torch.Tensor, k_p_r_max: float, grid: torch.Tensor):
    _min = torch.tensor([1e-12])
    pi4 = torch.pi * 4

    _kd = 2 * torch.pi * k_p_r_max * \
        torch.sqrt(torch.sum((
            torch.add(grid, displacement_vector[:,None,None,None])**2
        ), dim=0))
    res = pi4 * (torch.sin(_kd) - _kd * torch.cos(_kd))/torch.max(_min, _kd**3)
    res[torch.abs(_kd) <= 1e-12] = pi4/3
    return res


def _compute_reference(delta_: torch.Tensor, k_p_r_max: float, n_x: int, radius: float, centered: bool):
    if centered:
        axis = torch.linspace(-radius, radius, n_x, dtype=torch.float32)
    else:
        axis = torch.linspace(-radius, radius, n_x + 1, dtype=torch.float32)[:-1]
    (x2, x1, x0) = torch.meshgrid(axis, axis, axis, indexing='ij')
    grid = torch.stack([x0, x1, x2])

    # Note that I've confirmed that this version is equivalent to the one from the model.
    ref = torch.zeros((n_x, n_x, n_x), dtype=torch.float32)
    for d in delta_:
        ref += _planewave_integral(d, k_p_r_max, grid) * k_p_r_max**3
    return ref


def _get_spharm_volume(k_int: int, k_p_r_max: float, k_eq_d: float, n_x: int, radius: float, centered: bool):
    # NOTE: this is copied from various parts of
    # https://github.com/adirangan/dir_empm/blob/main/test/dir_interaction_required/test_convert_spharm_to_x_c_uniform_over_n_k_p_r_5.py
    delta_a_c_3s__ = torch.tensor([
        [1.5, -.5, .3],
        [-.5, -1.5, 2.]
    ], dtype=torch.float32) / (2 * k_p_r_max)

    kpr_max_f = float64(k_p_r_max)
    keqd_f = float64(k_eq_d)
    # (n_k_p_r, k_p_r_) = sample_sphere_7(0, kpr_max_f, keqd_f, 'L', 1, 0)[7:9]
    (n_k_p_r, k_p_r_, weight_3d_k_p_r_) = sample_sphere_7(0, kpr_max_f, keqd_f, 'L', 1, 0)[7:10]

    l_max_upb = k_int   # source rounds 2*pi*kpr_max, but kpr_max is k_int / 2pi
    l_max_ = torch.zeros(n_k_p_r, dtype=torch.int32)
    for nk_p_r in range(n_k_p_r):
        l_max_[nk_p_r] = max(0, min(l_max_upb, 1 + int(ceil(2 * torch.pi * k_p_r_[nk_p_r]))))
    n_y_ = (l_max_ + 1) ** 2
    n_y_sum = int(torch.sum(n_y_))

    a_k_Y_form_ = torch.zeros(n_y_sum, dtype=torch.complex64)
    for d in delta_a_c_3s__:
        a_k_Y_form_ += plane_wave_expansion_1(n_k_p_r, k_p_r_, d, l_max_)

    # to make this an actual Volume we also need the l_max_
    vol = Volume(l_max_, a_k_Y_form_)

    reference = _compute_reference(delta_a_c_3s__, k_p_r_max, n_x, radius, centered)

    (backup, _, _, _, _, _, _) = convert_spharm_to_x_c_uniform_over_n_k_p_r_5(
        0,
        k_eq_d,
        n_k_p_r,
        k_p_r_,
        k_p_r_max,
        weight_3d_k_p_r_,
        l_max_,
        a_k_Y_form_,
        radius,
        n_x,
        None, None, None, None, None,
        0 if centered else 1
    )
    # Shapes match out: backup and raveled reference are both 262144
    # Note BACKUP is giving a Complex128, though the imaginary part is pretty close to 0
    # raise ValueError(f"backup shape: {backup.shape} ref shape: {reference.ravel().shape}")
    # raise ValueError(f"{torch.max(backup.imag)} vs scale of real: {torch.max(backup.real)}")
    # raise ValueError(f"{torch.min(backup.imag)} vs scale of real: {torch.min(backup.real)}")
    backup2 = backup.real.to(dtype=torch.float32)
    # assert_close(backup2, reference.ravel())

    return (vol, reference, backup2)


@mark.parametrize("centered", [True, False])
def test_from_spharm_volume(centered: bool):
    k_int = 16
    k_p_r_max = k_int / (2 * torch.pi)
    k_eq_d = .5 / torch.pi
    half_diameter_x = 1.0
    n_x = 64
    radius = 1.

    (spharm_vol, ref_cartesian, b) = _get_spharm_volume(k_int, k_p_r_max, k_eq_d, n_x, radius, centered)

    res = CartesianVolume.from_spharm_volume(
        volume = spharm_vol,
        k_p_r_max = k_p_r_max,
        k_eq_d = k_eq_d,
        half_diameter_x = half_diameter_x,
        n_x = n_x,
        use_centered = centered
    )
    # ref-cartesian is float32
    # res axuxxx is complex128 in both cases... hmm
    assert_close(res.a_x_u_xxx_.real.to(dtype=torch.float32), b)
    assert_close(res.a_x_u_xxx_.real.to(dtype=torch.float32), ref_cartesian.ravel())
