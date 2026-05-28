import torch
from torch import Tensor

from empm.parameters import Parameters
from empm.grids import PolarGrid, FTK
from empm.stacks import CTFCluster, CTF, ImageStack, Poses, force_isotropy, Volume
from empm.core import execute_empm

# TODO: The below entries should be configured in an incoming Poses object
# euler_polar_a_ini_M_=None,
# euler_azimu_b_ini_M_=None,
# euler_gamma_z_ini_M_=None,
# image_delta_x_acc_ini_M_=None, --> maps to acc
# image_delta_y_acc_ini_M_=None,

## TODO: At this point, the wrapper can probably be merged with the
# main empm loop & the preparatory stuff kicked up another level.

def empm_loop_wrapper(  # former tfpmut_wrap_6. Name to be further revised.
        parameter: Parameters,
        grid: PolarGrid,
        weight_3d_k_p_r_: Tensor,
        ctfs: CTF,
        images: ImageStack,
        volume: Volume,
        initial_poses: Poses | None = None,
        delta_sigma_base: float = 0.0, # TODO: could this be vector-valued?
        ftk: FTK | None = None
):
    parameter.print_per_verbosity(" % [entering empm loop wrapper]")

    ctfs.expand_single_value_isotropic_ctf(grid) # TODO: does this need to be gated?
    if parameter.rank_CTF < 0:
        parameter.set_empirical_ctf_rank(ctfs, grid, images)

    if (parameter.flag_save_stage > 1):
        _checkpoint_initial(parameter, grid, volume)

    # TODO: This should probably be done when the parameter object is created
    # to avoid running the risk of ever resetting the seed midway
    if parameter.rseed is not None:
        torch.manual_seed(parameter.rseed)

    if (parameter.flag_clump_vs_cluster == 0):
        ctf_clusters = _cluster_ctfs(parameter, grid, ctfs, images, volume, delta_sigma_base)
    elif parameter.flag_clump_vs_cluster == 1:
        raise NotImplementedError("EMPM without clustering CTFs is not yet supported.")
    else:
        raise NotImplementedError("Flag clump vs cluster must be either 0 or 1")

    if initial_poses is None:
        initial_poses = Poses.random_init(n_imgs = images.n_M)

    if (parameter.flag_save_stage > 1):
        _checkpoint_poses(parameter, grid, volume, ctf_clusters, images, initial_poses, weight_3d_k_p_r_)

    execute_empm(
        parameter,
        grid,
        volume,
        images,
        ctf_clusters,
        initial_poses,
        weight_3d_k_p_r_,
        ftk = ftk
    )
    parameter.print_per_verbosity(" % [finished empm loop wrapper]")

    return volume


def _cluster_ctfs(
        params: Parameters,
        grid: PolarGrid,
        ctfs: CTF,
        images: ImageStack,
        volume: Volume,
        delta_sigma_base: float = 0.0
    ) -> CTFCluster:
    clusters = CTFCluster(params, grid, ctfs)
    if (params.flag_save_stage > 1):
        _checkpoint_clustering(params, clusters, grid)

    clusters.determine_principal_modes(params, grid, images, volume, delta_sigma_base)

    if (params.flag_save_stage > 1):
        clusters.checkpoint_report_clustering(params)
    
    return clusters


def _checkpoint_initial(param: Parameters, grid: PolarGrid, vol: Volume):
    param.save_report(
        name_tail = "_stage_0.mat",
        data = {
            "flag_verbose": param.flag_verbose,
            "tolerance_master": param.tolerance_master,
            "flag_gpu": param.flag_gpu,
            "flag_rank_vs_tolerance": param.flag_rank_vs_tolerance,
            "flag_clump_vs_cluster": param.flag_clump_vs_cluster,
            "tolerance_cluster": param.tolerance_cluster,
            "tolerance_pm": param.tolerance_pm,
            "rank_pm": param.rank_pm,
            "rank_CTF": param.rank_CTF,
            "rseed": param.rseed,
            "delta_r_max": param.delta_r_max,
            "n_iteration": param.n_iteration,
            "flag_alternate_MS_vs_SM": param.flag_alternate_MS_vs_SM,
            "n_w_": grid.n_w_,
            "n_w_sum": grid.n_w_sum,
            "n_w_max": grid.n_w_max,
            "n_w_csum_": grid.n_w_csum_,
            "l_max_max": vol.l_max_max,
            "n_y_": vol.n_y_,
            "n_y_max": vol.n_y_max,
            "n_y_sum": vol.n_y_sum,
            "n_y_csum_": vol.n_y_csum_,
        },
        min_level = 1
    )


def _checkpoint_clustering(param: Parameters, cluster: CTFCluster, grid: PolarGrid):
    param.save_report(
        name_tail = "_stage_1.mat",
        data = {
            "index_ncluster_from_nCTF_": cluster.index_ncluster_from_nCTF_,
            "n_k_p_r": grid.n_k_p_r,
            "k_p_r_": grid.k_p_r_,
            "weight_2d_k_p_r_": grid.weight_2d_k_p_r_,
            "n_CTF": cluster.ctfs.n_CTF,
            "CTF_k_p_r_kC__":  force_isotropy(cluster.ctfs, grid),
            "index_nM_from_ncluster__":  cluster.index_nM_from_ncluster__,
            "n_index_nM_from_ncluster_":  cluster.n_index_nM_from_ncluster_,
        },
        min_level = 1
    )


def _checkpoint_poses(
    param: Parameters,
    grid: PolarGrid,
    vol: Volume,
    ctf_cluster: CTFCluster,
    images: ImageStack,
    poses: Poses,
    weight_3d_k_p_r_: Tensor
):
    # Is it really necessary to checkpoint all this stuff again?
    # We already have most of it
    param.save_report(
        name_tail = "_stage_3.mat",
        data = {
            "n_k_p_r": grid.n_k_p_r,
            "k_p_r_": grid.k_p_r_,
            "k_p_r_max": grid.k_p_r_max,
            "weight_3d_k_p_r_": weight_3d_k_p_r_,
            "weight_2d_k_p_r_": grid.weight_2d_k_p_r_,
            "n_w_": grid.n_w_,
            "weight_2d_k_p_wk_": grid.weight_2d_k_p_wk_,
            "l_max_": vol.l_max_,
            "n_CTF": ctf_cluster.ctfs.n_CTF,
            "CTF_k_p_wkC__": ctf_cluster.ctfs.CTF_k_p_wkC__,
            "index_nCTF_from_nM_": ctf_cluster.ctfs.index_nCTF_from_nM_,
            "n_M": images.n_M,
            "M_k_p_wkM__": images.M_k_p_wkM__,
            "n_cluster": ctf_cluster.n_cluster,
            "index_ncluster_from_nCTF_": ctf_cluster.index_ncluster_from_nCTF_,
            "pm_n_UX_rank_c_": ctf_cluster.pm_n_UX_rank_c_,
            "pm_UX_knc___": ctf_cluster.pm_UX_knc___,
            "pm_X_weight_rc__": ctf_cluster.pm_X_weight_rc__,
            "euler_polar_a_M_": poses.euler_polar_a_M_,
            "euler_azimu_b_M_": poses.euler_azimu_b_M_,
            "euler_gamma_z_M_": poses.euler_gamma_z_M_,
            "image_delta_x_acc_M_": poses.image_delta_x_acc_M_,
            "image_delta_y_acc_M_": poses.image_delta_y_acc_M_,
        },
        min_level= 1
    )
