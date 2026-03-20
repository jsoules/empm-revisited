from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from torch import Tensor

from dir_empm.qbp_uniform_over_n_k_p_r_10 import qbp_uniform_over_n_k_p_r_10
from dir_empm.pm_template_3 import pm_template_3
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_

from empm.parameters import Parameters
from empm.grids import PolarGrid
from empm.util import zero_initial_csum
from .Templates import Templates

if TYPE_CHECKING:
    from .CTFCluster import CTFCluster
    from .ImageStack import ImageStack
    from .Poses import Poses

# FYI
# # def qbp_uniform_over_n_k_p_r_10(
# #         qbp_eps=None,               # tolerance
# #         n_k_p_r=None,               # from grid
# #         k_p_r_=None,                # from grid
# #         l_max_=None,                # from spherical harmonics
# #         n_w_=None,                  # from grid
# #         n_M=None,                   # from image stack
# #         M_k_p_wkM__=None,           # is the image stack
# #         index_nCTF_from_nM_=None,   # from CTFs
# #         CTF_k_p_wkC__=None,         # is the CTFs
# #         euler_polar_a_M_=None,      # from poses
# #         euler_azimu_b_M_=None,      # rest from poses
# #         euler_gamma_z_M_=None,
# #         image_delta_x_M_=None,
# #         image_delta_y_M_=None,
# #         image_I_value_M_=None,
# # ) -> tuple[Tensor, Tensor, Tensor]:
# #     # These are supposed to be a_k_Y_yk__, n_quad_from_data_q_, a_k_p_qk__.ravel()
# #     ...


# # def pm_template_3(
# #     flag_verbose=None,
# #     l_max=None,
# #     n_k=None,
# #     a_k_Y_yk__=None,
# #     viewing_euler_k_eq_d=None,
# #     template_inplane_k_eq_d=None,
# #     n_w_input=None,
# #     n_S=None,
# #     viewing_azimu_b_S_=None,
# #     viewing_polar_a_S_=None,
# #     viewing_weight_S_=None,
# #     n_viewing_polar_a=None,
# #     viewing_polar_a_=None,
# #     n_viewing_azimu_b_=None,
# #     sqrt_2lp1_=None,
# #     sqrt_2mp1_=None,
# #     sqrt_rat0_m_=None,
# #     sqrt_rat3_lm__=None,
# #     sqrt_rat4_lm__=None,
# # ) -> tuple[Tensor, int, int, Tensor, Tensor, Tensor, int, Tensor, int, Tensor, Tensor, Tensor, Tensor, Tensor]:
# #     ...
# #         # template_wkS___,
# #         # n_w,
# #         # n_S,
# #         # viewing_azimu_b_S_,
# #         # viewing_polar_a_S_,
# #         # viewing_weight_S_,
# #         # n_viewing_polar_a,
# #         # viewing_polar_a_,
# #         # n_viewing_azimu_b_,
# #         # sqrt_2lp1_,
# #         # sqrt_2mp1_,
# #         # sqrt_rat0_m_,
# #         # sqrt_rat3_lm__,
# #         # sqrt_rat4_lm__,


# # # This can probably be brought in-house
# # def local_yk__from_yk_(
# #     n_k_p_r: int,
# #     l_max_: Tensor,
# #     tmp_yk_: Tensor,
# # ) -> tuple[Tensor]:
# #     ...

######

class Volume():
    """Class representing a volume using a spherical harmonic basis.

    Attributes:
        l_max_ (Tensor): Maximum order of spherical harmonic something or other.
            Should be integer-valued 1d tensor, per-something.
        l_max_max (int): Maximum of any order in l_max_
        n_y_ (Tensor): TODO
        n_y_max (int): TODO
        n_y_sum (int): TODO
        n_y_csum_ (Tensor): TODO
        a_k_Y_reco_yk_ (Tensor): TODO
    """

    l_max_: Tensor
    l_max_max: int
    n_y_: Tensor
    n_y_max: int
    n_y_sum: int
    n_y_csum_: Tensor
    a_k_Y_reco_yk_: Tensor

    def __init__(self, l_max_: Tensor):
        # TODO: representation checking (dtype, shape, etc)
        self.l_max_ = l_max_
        self.l_max_max = int(torch.max(l_max_).item())
        self.n_y_ = (l_max_+1) ** 2
        self.n_y_max = int(torch.max(self.n_y_).item())
        self.n_y_sum = int(torch.sum(self.n_y_).item())
        self.n_y_csum_ = zero_initial_csum(self.n_y_)
        self.a_k_Y_reco_yk_ = torch.zeros(1, dtype=torch.complex64)


    def reconstruct_volume(self,
        parameter: Parameters,
        grid: PolarGrid,
        image_stack: ImageStack,
        poses: Poses,
        ctf_cluster: CTFCluster,
        weight_3d_k_p_r_: Tensor,       # TODO: can we link this to the spherical harmonic object?
        niteration: int
    ):
        # "use current euler-angles and displacements to solve for current model."
        qbp_eps = parameter.tolerance_master
        self.a_k_Y_reco_yk_ = qbp_uniform_over_n_k_p_r_10(
            qbp_eps,
            grid.n_k_p_r,
            grid.k_p_r_,
            self.l_max_,
            grid.n_w_,
            image_stack.n_M,
            image_stack.M_k_p_wkM__,
            ctf_cluster.ctfs.index_nCTF_from_nM_,
            ctf_cluster.ctfs.CTF_k_p_wkC__,
            poses.euler_polar_a_M_,
            poses.euler_azimu_b_M_,
            poses.euler_gamma_z_M_,
            poses.image_delta_x_acc_M_ + poses.image_delta_x_upd_M_,
            poses.image_delta_y_acc_M_ + poses.image_delta_y_upd_M_,
        )[0]

        # TODO: this gets persisted before we update the model?
        if (parameter.flag_save_stage > 2):
            _checkpoint_iteration_model(parameter, niteration, True, self.a_k_Y_reco_yk_)

        #% Normalize a_k_Y_reco_yk_ to prevent the intensity from diverging over successive iterations
        self._spharm_normalize(weight_3d_k_p_r_)

        if (parameter.flag_save_stage>2):
            _checkpoint_iteration_model(parameter, niteration, False, self.a_k_Y_reco_yk_)


    def _spharm_normalize(self,
        weight_3d_k_p_r_: Tensor,  # TODO put on this object etc
    ) -> None:
        # Normalizes the spherical-harmonic expansion to have norm 1, but does not center.
        # expand the per-ring weights to the number of y-elements in each radial ring.
        weight_Y_val_ = weight_3d_k_p_r_.repeat_interleave(self.n_y_)

        a_std = torch.linalg.vector_norm(self.a_k_Y_reco_yk_ * torch.sqrt(weight_Y_val_))
        denom = max(1e-12, a_std)
        self.a_k_Y_reco_yk_ = self.a_k_Y_reco_yk_ / denom
    

    def generate_templates(self,
        parameter: Parameters,
        grid: PolarGrid,
        niter: int,
    ) -> Templates:
        # TODO: Figure out how to reuse the memory allocated for the last iteration of templates
        # (requires constructing tensor of proper size the first time, and also making modifications
        # to pm_template_3)

        #% Construct templates using the volume.
        (
            S_k_p_wkS__,
            _,
            n_S,
            viewing_azimu_b_S_,
            viewing_polar_a_S_,
        ) = pm_template_3(
            0, # flag-verbose is hard-coded off
            self.l_max_max,
            grid.n_k_p_r,
            torch.reshape(local_yk__from_yk_(grid.n_k_p_r, self.l_max_, self.a_k_Y_reco_yk_)[0], (grid.n_k_p_r, self.n_y_max)),
            parameter.template_viewing_k_eq_d,
            -1,
            grid.n_w_max,
        )[:5]
        S_k_p_wkS__ = torch.reshape(S_k_p_wkS__, (n_S, grid.n_w_sum))
        templates = Templates(n_S, S_k_p_wkS__, viewing_azimu_b_S_, viewing_polar_a_S_)

        if (parameter.flag_save_stage > 2):
            _checkpoint_templates(parameter, templates, niter)

        return templates
        


def _checkpoint_iteration_model(
    parameter: Parameters,
    niteration: int,
    is_pre: bool,
    a_k_Y_reco_yk_: Tensor
):
    if is_pre:
        fname = f"_stage_6_{niteration}.mat"
    else:
        fname = f"_stage_7_{niteration}.mat"
    parameter.save_report(fname, data = {
        "a_k_Y_reco_yk_": a_k_Y_reco_yk_
    })


def _checkpoint_templates(
    parameter: Parameters,
    templates: Templates,
    niteration: int
):
    parameter.save_report(f"_stage_8_{niteration}.mat", data = {
        "S_k_p_wkS__": templates.S_k_p_wkS__,
        "n_S": templates.n_S,
        "viewing_azimu_b_S_": templates.viewing_azimu_b_S_,
        "viewing_polar_a_S_": templates.viewing_polar_a_S_,
    })
