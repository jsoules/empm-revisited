import torch
from torch import Tensor

# TODO: Fix this for proper imports
from ..parameters import Parameters
from ..grids import PolarGrid, SphericalHarmonics, FTK
from ..stacks import CTFCluster, ImageStack, Poses, Templates, force_isotropy


# TODO: Proper import for tfpmh_Z_cluster_wrap_SM__14
# sig:
def tfpmh_Z_cluster_wrap_SM__14(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        n_w_=None,
        weight_2d_k_p_r_=None,
        weight_2d_k_p_wk_=None,
        n_S=None,
        S_k_p_wkS__=None,
        n_CTF=None,
        CTF_k_p_r_kC__=None,
        index_nCTF_from_nM_=None,
        n_M=None,
        M_k_p_wkM__=None,
        n_cluster=None,
        index_ncluster_from_nCTF_=None,
        pm_n_UX_rank_c_=None,
        pm_UX_knc___=None,
        pm_X_weight_rc__=None,
        FTK=None,
        index_nM_to_update_=torch.tensor([]).to(dtype=torch.int32),
        M_k_q_wkM__=None,
        UX_T_M_l2_dM__=None,
        UX_M_l2_M_=None,
        svd_V_UX_M_lwnM____=None, #%<-- or UX_T_M_k_q_dwnM____ ;
        index_nS_to_update_=torch.tensor([]).to(dtype=torch.int32),
        UX_CTF_S_k_q_wnS__=None,
        UX_CTF_S_l2_S_=None,
) -> tuple[dict, Tensor, Tensor, Tensor, 
           Tensor, Tensor, Tensor, Tensor, 
           Tensor, Tensor, Tensor, Tensor, 
           Tensor, Tensor, Tensor, Tensor ]:
    #     return(
    #     parameter,
    #     Z_SM__,
    #     UX_CTF_S_l2_SM__,
    #     UX_T_M_l2_SM__,
    #     X_SM__,
    #     delta_x_SM__,
    #     delta_y_SM__,
    #     gamma_z_SM__,
    #     index_sub_SM__,
    #     index_nM_from_ncluster__,
    #     n_index_nM_from_ncluster_,
    #     M_k_q_wkM__,
    #     UX_T_M_l2_dM__,
    #     UX_M_l2_M_,
    #     svd_V_UX_M_lwnM____, #%<-- or UX_T_M_k_q_dwnM____ ;
    #     UX_CTF_S_k_q_wnS__,
    # );
    ...

## TODO: Proper import for tfpmh_MS_vs_SM_2
# declaration:
def tfpmh_MS_vs_SM_2(
        parameter =None,
        n_w_max=None,
        n_S=None,
        viewing_azimu_b_S_=None,
        viewing_polar_a_S_=None,
        n_M=None,
        X_SM__=None,
        delta_x_SM__=None,
        delta_y_SM__=None,
        gamma_z_SM__=None,
        I_value_SM__=None,
) -> tuple[dict,
           Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor, Tensor, Tensor]:
        # parameter,
        # euler_polar_a_M_,
        # euler_azimu_b_M_,
        # euler_gamma_z_M_,
        # image_delta_x_M_,
        # image_delta_y_M_,
        # image_I_value_M_,
        # image_X_value_M_,
        # image_S_index_M_,
    ...


class AlignmentScratch():
    M_k_q_wkM__         : Tensor | None
    UX_T_M_l2_dM__      : Tensor | None
    UX_M_l2_M_          : Tensor | None
    svd_V_UX_M_lwnM____ : Tensor | None
    UX_CTF_S_k_q_wnS__  : Tensor | None
    UX_CTF_S_l2_S_      : Tensor | None
    Z_SM__              : Tensor | None
    UX_T_M_l2_SM__      : Tensor | None
    delta_x_SM__        : Tensor | None
    delta_y_SM__        : Tensor | None
    gamma_z_SM__        : Tensor | None
    index_sub_SM__      : Tensor | None

    def __init__(self):
        self.M_k_q_wkM__ = None
        self.UX_T_M_l2_dM__ = None
        self.UX_M_l2_M_ = None
        self.svd_V_UX_M_lwnM____ = None
        self.UX_CTF_S_k_q_wnS__ = None
        self.UX_CTF_S_l2_S_ = None
        self.Z_SM__ = None
        self.UX_T_M_l2_SM__ = None
        self.X_SM__ = None
        self.delta_x_SM__ = None
        self.delta_y_SM__ = None
        self.gamma_z_SM__ = None
        self.index_sub_SM__ = None


class Alignment():

    parameter: Parameters
    grid: PolarGrid
    harmonics: SphericalHarmonics
    ctf_clusters: CTFCluster
    image_stack: ImageStack
    ftk: FTK
    poses: Poses
    scratch: AlignmentScratch
    X_SM__: Tensor | None

    def __init__(self,
        parameter: Parameters,
        grid: PolarGrid,
        harmonics: SphericalHarmonics,
        ctf_clusters: CTFCluster,
        image_stack: ImageStack,
        ftk: FTK,
        poses: Poses
    ):
        self.parameter = parameter
        self.grid = grid
        self.harmonics = harmonics
        self.ctf_clusters = ctf_clusters
        self.image_stack = image_stack
        self.ftk = ftk
        self.poses = poses
        self.scratch = AlignmentScratch()
        self.X_SM__ = None


    # NOTE: templates could be added to the object rather than passed in,
    # if we can arrange for it to be a reusable arena with a constant object
    # reference rather than getting overwritten at each cycle
    def align(self, templates: Templates, M_pert_k_p_wkM__: Tensor, niter: int):
        # Do alignment from volume, grouping principal-images by cluster.
        # (NOTE: The below should really be a docstring for tfpmh_Z_cluster_wrap_SM__14.)
        #% Use given volume to align principal-images. ;
        #% Groups principal-images by cluster. ;
        #% Calculates principal-templates associated with each cluster. ;
        #% Uses precomputation for M and S. ;
        #% Batches images into batches of size n_M_per_Mbatch (default 24). ;
        #% Batches templates into batches of size n_S_per_Sbatch (default 24). ;
        #% Only stores the optimal translation for each principal-image. ;
        self.parameter.reduce_verbosity()
        # TODO: torch.ones, surely?
        index_nS_to_update_ = torch.arange(templates.n_S).to(dtype=torch.int32)
        if self.parameter.flag_precompute_UX_CTF_S_k_q_wnS__ == 0 and self.parameter.flag_precompute_UX_CTF_S_l2_S_ == 0:
            index_nS_to_update_ = torch.arange(0, dtype=torch.int32)

        (
            _,  # unused parameter return
            self.scratch.Z_SM__,
            self.scratch.UX_CTF_S_l2_S_,
            self.scratch.UX_T_M_l2_SM__,  # only used for reporting
            self.X_SM__,
            self.scratch.delta_x_SM__,
            self.scratch.delta_y_SM__,
            self.scratch.gamma_z_SM__,
            self.scratch.index_sub_SM__,     # only used for reporting
            _, #index_nM_from_ncluster__,   # only used for reporting -- QUERY: is this actually allowed to change??
            _, #n_index_nM_from_ncluster_,  # only used for reporting -- QUERY: is this actually allowed to change??
            self.scratch.M_k_q_wkM__,
            self.scratch.UX_T_M_l2_dM__,
            self.scratch.UX_M_l2_M_,
            self.scratch.svd_V_UX_M_lwnM____,
            self.scratch.UX_CTF_S_k_q_wnS__,
        ) = tfpmh_Z_cluster_wrap_SM__14(
            self.parameter.to_dict(),
            self.grid.n_k_p_r,
            self.grid.k_p_r_,
            self.grid.k_p_r_max,
            self.grid.n_w_,
            self.grid.weight_2d_k_p_r_,
            self.grid.weight_2d_k_p_wk_,
            templates.n_S,
            templates.S_k_p_wkS__,
            self.ctf_clusters.ctfs.n_CTF,
            force_isotropy(self.ctf_clusters.ctfs, self.grid), # aka CTF_k_p_r_kC__,
            self.ctf_clusters.ctfs.index_nCTF_from_nM_,
            self.image_stack.n_M,
            M_pert_k_p_wkM__,
            self.ctf_clusters.n_cluster,
            self.ctf_clusters.index_ncluster_from_nCTF_,
            self.ctf_clusters.pm_n_UX_rank_c_,
            self.ctf_clusters.pm_UX_knc___,
            self.ctf_clusters.pm_X_weight_rc__,
            self.ftk,
            torch.where(self.poses.flag_image_delta_upd_M_)[0], # aka index_nM_to_update_,
            self.scratch.M_k_q_wkM__,
            self.scratch.UX_T_M_l2_dM__,
            self.scratch.UX_M_l2_M_,
            self.scratch.svd_V_UX_M_lwnM____,
            index_nS_to_update_,
            self.scratch.UX_CTF_S_k_q_wnS__,
            self.scratch.UX_CTF_S_l2_S_,
        )[:16]
        self.parameter.restore_verbosity()

        if (self.parameter.flag_save_stage > 2):
            self._checkpoint_post_alignment(templates, M_pert_k_p_wkM__, niter)


    def update_poses_from_alignment(self, templates: Templates, niter: int):
        # update poses from alignment
        self.parameter.reduce_verbosity()
        (
            _,
            # TODO: Assigning directly to an object's member variables feels wrong, also are we
            # duplicitavely reallocating memory? Need to look at internals
            self.poses.euler_polar_a_M_,
            self.poses.euler_azimu_b_M_,
            self.poses.euler_gamma_z_M_,
            self.poses.image_delta_x_bit_M_,
            self.poses.image_delta_y_bit_M_,
            self.poses.image_I_value_M_,
            image_X_value_M_,   # only for reporting
            image_S_index_M_,   # only for reporting
        ) = tfpmh_MS_vs_SM_2(
            self.parameter.to_dict(),
            self.grid.n_w_max,
            templates.n_S,
            templates.viewing_azimu_b_S_,
            templates.viewing_polar_a_S_,
            self.image_stack.n_M,
            self.X_SM__,
            self.scratch.delta_x_SM__,
            self.scratch.delta_y_SM__,
            self.scratch.gamma_z_SM__,
        )[:9]
        self.parameter.restore_verbosity()


        if (self.parameter.flag_save_stage > 2):
            self._checkpoint_pose_update(image_X_value_M_, image_S_index_M_, niter)

        ...


    def _checkpoint_post_alignment(self,
        templates: Templates,
        M_pert_k_p_wkM__: Tensor,
        niter: int
    ):
        self.parameter.save_report(f"_stage_9_{niter}.mat", data = {
        "n_k_p_r": self.grid.n_k_p_r,
        "k_p_r_": self.grid.k_p_r_,
        "k_p_r_max": self.grid.k_p_r_max,
        "n_w_": self.grid.n_w_,
        "weight_2d_k_p_r_":  self.grid.weight_2d_k_p_r_,
        "weight_2d_k_p_wk_": self.grid.weight_2d_k_p_wk_,
        "n_S": templates.n_S,
        "S_k_p_wkS__": templates.S_k_p_wkS__,
        "n_CTF": self.ctf_clusters.ctfs.n_CTF,
        "CTF_k_p_r_kC__": force_isotropy(self.ctf_clusters.ctfs, self.grid),
        "index_nCTF_from_nM_": self.ctf_clusters.ctfs.index_nCTF_from_nM_,
        "n_M": self.image_stack.n_M,
        "M_pert_k_p_wkM__": M_pert_k_p_wkM__,
        "n_cluster": self.ctf_clusters.n_cluster,
        "index_ncluster_from_nCTF_": self.ctf_clusters.index_ncluster_from_nCTF_,
        "pm_n_UX_rank_c_": self.ctf_clusters.pm_n_UX_rank_c_,
        "pm_UX_knc___": self.ctf_clusters.pm_UX_knc___,
        "pm_X_weight_rc__": self.ctf_clusters.pm_X_weight_rc__,
        "Z_SM__": self.scratch.Z_SM__,
        "UX_CTF_S_l2_S_": self.scratch.UX_CTF_S_l2_S_,
        "UX_T_M_l2_SM__": self.scratch.UX_T_M_l2_SM__,
        "X_SM__": self.X_SM__,
        "delta_x_SM__": self.scratch.delta_x_SM__,
        "delta_y_SM__": self.scratch.delta_y_SM__,
        "gamma_z_SM__": self.scratch.gamma_z_SM__,
        "index_sub_SM__": self.scratch.index_sub_SM__,
        "index_nM_from_ncluster__": self.ctf_clusters.index_nM_from_ncluster__,
        "n_index_nM_from_ncluster_": self.ctf_clusters.n_index_nM_from_ncluster_,
        # "M_pert_k_p_wkM__": M_pert_k_p_wkM__, # this was a repeat
        "M_k_q_wkM__": self.scratch.M_k_q_wkM__,
        "UX_T_M_l2_dM__": self.scratch.UX_T_M_l2_dM__,
        "UX_M_l2_M_": self.scratch.UX_M_l2_M_,
        "svd_V_UX_M_lwnM____": self.scratch.svd_V_UX_M_lwnM____,
        "UX_CTF_S_k_q_wnS__": self.scratch.UX_CTF_S_k_q_wnS__,
    })


    def _checkpoint_pose_update(self,
        image_X_value_M_: Tensor,
        image_S_index_M_: Tensor,
        niteration: int,
    ):
        self.parameter.save_report(f"_stage_10_{niteration}.mat", data = {
            "euler_polar_a_M_": self.poses.euler_polar_a_M_,
            "euler_azimu_b_M_": self.poses.euler_azimu_b_M_,
            "euler_gamma_z_M_": self.poses.euler_gamma_z_M_,
            "image_delta_x_bit_M_": self.poses.image_delta_x_bit_M_,
            "image_delta_y_bit_M_": self.poses.image_delta_y_bit_M_,
            "image_I_value_M_": self.poses.image_I_value_M_,
            "image_X_value_M_": image_X_value_M_,
            "image_S_index_M_": image_S_index_M_,
        })

