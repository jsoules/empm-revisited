import torch
from torch import Tensor
from math import floor
from typing import Callable

from .parameters import Parameters, MACHINE_TOLERANCE
from .grids import PolarGrid, FTK
from .stacks import Alignment, CTFCluster, ImageStack, Poses, force_isotropy, Volume

# TODO: THIS AFFECTS CALLERS
# TODO NOTE: When setting up parameters, we want r8_svd_eps to be TOLERANCE_MASTER not its default value
# TODO NOTE: When setting up parameters, we want r8_delta_r_max to be delta_r_max, NOT 0.0
# NOTE: Ensure CTF.expand_single_value_isotropic_ctf() has already been called!
# Presumably that would be much higher in the call stack

# NOTE: if CTF_k_p_r_kC__ is needed, can be obatined from ctf_cluster.foce_isotropy()
# NOTE: Checkpointing precomputation is still expensive, even if we are skipping the actual output.

def execute_empm(       # replaces tfpmut_6
    parameter: Parameters,
    grid: PolarGrid,
    volume: Volume,
    image_stack: ImageStack,
    ctf_cluster: CTFCluster,
    poses: Poses,
    weight_3d_k_p_r_: Tensor,       # TODO: can we link this to the spherical harmonic object?
    alignment: Alignment | None = None,
    ftk: FTK | None = None,
):
    parameter.print_per_verbosity(f' %% [entering execute_empm, former tfpmut_6]')
    ctf_cluster.make_cluster_averages(grid) # should be pleonastic--it's called during cluster construction
    M_pert_k_p_wkM__ = None

    if ftk is None:
        ftk = FTK.from_grid(grid, parameter)

    if alignment is None:
        alignment = Alignment(parameter, grid, volume, ctf_cluster, image_stack, ftk, poses)

    # This should have been seeded higher up if it is required
    # Otherwise we'll be resetting the seed
    # if parameter.rseed is not None:
    #     torch.manual_seed(parameter.rseed)

    if parameter.flag_save_stage > 2:
        _checkpoint_preloop(parameter, grid, volume, ctf_cluster, poses, ftk)

    for niteration in range(parameter.n_iteration):
        parameter.print_per_verbosity(f" %% niteration {niteration:.2d}/{parameter.n_iteration:.2d}")

        M_pert_k_p_wkM__ = image_stack.apply_displacements_from_poses(grid, poses, M_pert_k_p_wkM__)
        if parameter.flag_save_stage > 2:
            _checkpoint_iteration_displacement(parameter, niteration, poses, image_stack, M_pert_k_p_wkM__)

        volume.reconstruct_volume(parameter, grid, image_stack, poses, ctf_cluster, weight_3d_k_p_r_, niteration)
        templates = volume.generate_templates(parameter, grid, niteration)
        alignment.align(templates, M_pert_k_p_wkM__, niteration)
        alignment.update_poses_from_alignment(templates, niteration)
        poses.update_translations(parameter, MACHINE_TOLERANCE)

        if (parameter.flag_save_stage > 2):
            _checkpoint_disp_update(parameter, poses, niteration)

    parameter.print_per_verbosity(f' %% [finished execute_empm, former tfpmut_6]')

    return (volume, poses)


def _checkpoint_preloop(
    parameter: Parameters,
    grid: PolarGrid,
    harmonics: Volume,
    ctf_cluster: CTFCluster,
    poses: Poses,
    ftk: FTK
):
    # TODO: simplify this further
    parameter.save_report("_stage_4.mat", data = {
        **parameter.to_dict(),
        "svd_eps": parameter.tolerance_master,
        "n_w_": grid.n_w_,
        "n_w_sum": grid.n_w_sum,
        "n_w_max": grid.n_w_max,
        "n_w_csum_": grid.n_w_csum_,
        "l_max_max": harmonics.l_max_max,
        "n_y_": harmonics.n_y_,
        "n_y_max": harmonics.n_y_max,
        "n_y_sum": harmonics.n_y_sum,
        "n_y_csum_": harmonics.n_y_csum_,
        "n_cluster": ctf_cluster.n_cluster,
        "index_ncluster_from_nM_": ctf_cluster.index_ncluster_from_nM_,
        "index_nM_from_ncluster__": ctf_cluster.index_nM_from_ncluster__,
        "n_index_nM_from_ncluster_": ctf_cluster.n_index_nM_from_ncluster_,
        "CTF_k_p_wkC__": ctf_cluster.ctfs.CTF_k_p_wkC__,
        "n_CTF": ctf_cluster.ctfs.n_CTF,
        "CTF_k_p_r_kC__": force_isotropy(ctf_cluster.ctfs, grid),
        # "CTF_k_p_wkM__": ctf_cluster.ctfs.CTF_k_p_wkM__,   # I forget what this is
        "CTF_k_p_r_xavg_kc__": ctf_cluster.CTF_k_p_r_xavg_kc__,
        "n_delta_v": ftk.n_delta_v,
        "n_svd_l": ftk.n_svd_l,
        "euler_polar_a_M_": poses.euler_polar_a_M_,
        "euler_azimu_b_M_": poses.euler_azimu_b_M_,
        "euler_gamma_z_M_": poses.euler_gamma_z_M_,
        "image_delta_x_acc_M_": poses.image_delta_x_acc_M_,
        "image_delta_y_acc_M_": poses.image_delta_y_acc_M_,
        "image_delta_x_upd_M_": poses.image_delta_x_upd_M_,
        "image_delta_y_upd_M_": poses.image_delta_y_upd_M_,
        "image_delta_x_bit_M_": poses.image_delta_x_bit_M_,
        "image_delta_y_bit_M_": poses.image_delta_y_bit_M_,
        "flag_image_delta_upd_M_": poses.flag_image_delta_upd_M_,
        "image_I_value_M_": poses.image_I_value_M_,
    })


def _checkpoint_iteration_displacement(
    parameter: Parameters,
    niteration: int,
    poses: Poses,
    image_stack: ImageStack,
    M_pert_k_p_wkM__: Tensor,
):
    parameter.save_report(f"_stage_5_{niteration}.mat", data = {
                "index_nM_to_update_": torch.where(poses.flag_image_delta_upd_M_)[0],
                "M_pert_k_p_wkM__": M_pert_k_p_wkM__,
                "M_orig_k_p_wkM__": image_stack.M_k_p_wkM__,
                "image_delta_x_acc_M_": poses.image_delta_x_acc_M_,
                "image_delta_y_acc_M_": poses.image_delta_y_acc_M_,
            })


def _checkpoint_disp_update(
    parameter: Parameters,
    poses: Poses,
    niteration: int
):
    parameter.save_report(f"_stage_11_{niteration}.mat", data = {
        "delta_r_upd_threshold": parameter.delta_r_upd_threshold,
        "flag_image_delta_upd_M_": poses.flag_image_delta_upd_M_,
        "image_delta_x_acc_M_": poses.image_delta_x_acc_M_,
        "image_delta_y_acc_M_": poses.image_delta_y_acc_M_,
        "image_delta_x_upd_M_": poses.image_delta_x_upd_M_,
        "image_delta_y_upd_M_": poses.image_delta_y_upd_M_,
    })

