from torch import Tensor
import torch
from numpy import float64

from .Volume import Volume
from empm.util import generate_equispaced_points

from dir_empm.sample_sphere_7 import sample_sphere_7
from dir_empm.convert_spharm_to_k_p_uniform_over_n_k_p_r_5 import convert_spharm_to_k_p_uniform_over_n_k_p_r_5
from dir_empm.xxnufft3d3 import xxnufft3d3


# TODO: more transparent names
# TODO: This class, like the (implicitly spherical-harmonic) Volume class before it,
# combines a spherical grid (the address space in which the data lives) with the
# data field itself. For our purposes thus far this isn't a problem because it doesn't
# make sense for us to use one of these objects without the data also existing, but
# this will be an issue if we try to use them in a broader context in the future.
# SO TODO: Split into separate sphere-grid and volume classes.

class CartesianVolume():
    """Class holding Cartesian real-space representation of a
    reconstructed volume.

    Attributes:
        a_x_u_xxx_ (Tensor): TKTK
        a_k_p_qk_ (Tensor): TKTK
        sqrt_2lp1_ (Tensor): TKTK
        sqrt_2mp1_ (Tensor): TKTK
        sqrt_rat0_m_ (Tensor): TKTK
        sqrt_rat3_lm__ (Tensor): TKTK
        sqrt_rat4_lm__ (Tensor): TKTK
    """

    a_x_u_xxx_: Tensor
    a_k_p_qk_: Tensor
    sqrt_2lp1_: Tensor
    sqrt_2mp1_: Tensor
    sqrt_rat0_m_: Tensor
    sqrt_rat3_lm__: Tensor
    sqrt_rat4_lm__: Tensor


    def __init__(self, *,
        a_x_u_xxx_: Tensor,
        a_k_p_qk_: Tensor,
        sqrt_2lp1_: Tensor,
        sqrt_2mp1_: Tensor,
        sqrt_rat0_m_: Tensor,
        sqrt_rat3_lm__: Tensor,
        sqrt_rat4_lm__: Tensor,
    ):
        self.a_x_u_xxx_ = a_x_u_xxx_
        self.a_k_p_qk_ = a_k_p_qk_
        self.sqrt_2lp1_ = sqrt_2lp1_
        self.sqrt_2mp1_ = sqrt_2mp1_
        self.sqrt_rat0_m_ = sqrt_rat0_m_
        self.sqrt_rat3_lm__ = sqrt_rat3_lm__
        self.sqrt_rat4_lm__ = sqrt_rat4_lm__


    # TODO: Avoid hard-coded precision?
    @classmethod
    def from_spharm_volume(cls,
        volume: Volume,
        k_p_r_max: float = 48 / (2 * torch.pi),
        k_eq_d: float = 1. / (2 * torch.pi),
        half_diameter_x_c: float = 1.0,
        n_x_u_pack: int = 64,
    ) -> CartesianVolume:
        _axis_points = generate_equispaced_points(half_diameter_x_c, n_x_u_pack)
        x_u_2, x_u_1, x_u_0 = torch.meshgrid(_axis_points, _axis_points, _axis_points, indexing='ij')
        n_xxx_u = n_x_u_pack ** 3
        a_x_u_xxx_ = torch.zeros(n_xxx_u, dtype=torch.complex64)

        (
            n_qk,
            n_qk_csum_,
            k_p_r_qk_,
            k_p_azimu_b_qk_,
            k_p_polar_a_qk_,
            weight_3d_k_p_qk_,
            weight_shell_qk_,
            n_k_p_r,
            k_p_r_,
            weight_3d_k_p_r_,
            k_c_0_qk_,
            k_c_1_qk_,
            k_c_2_qk_,
        ) = sample_sphere_7(
            flag_verbose = 0,
            k_p_r_max = float64(k_p_r_max),
            k_eq_d = float64(k_eq_d),
            str_T_vs_L = 'L',
            flag_uniform_over_n_k_p_r = 1
        )[:13]
        # ^-- sum(weight_3d_k_p_r_)*(4*pi) = (4/3)*pi*k_p_r_max^3 --> sum(weight_3d_k_p_r_) = (1/3)*k_p_r_max^3 ;

        (
            a_k_p_qk_,
            sqrt_2lp1_,
            sqrt_2mp1_,
            sqrt_rat0_m_,
            sqrt_rat3_lm__,
            sqrt_rat4_lm__,
        ) = convert_spharm_to_k_p_uniform_over_n_k_p_r_5(
            0,  # flag_verbose
            n_qk,
            n_qk_csum_,
            k_p_r_qk_,
            k_p_azimu_b_qk_,
            k_p_polar_a_qk_,
            weight_3d_k_p_qk_,
            weight_shell_qk_,
            n_k_p_r,
            k_p_r_,
            weight_3d_k_p_r_,
            volume.l_max_,
            volume.a_k_Y_reco_yk_,  # NOTE: remove "reco" from the Volume class name
            # sqrt_2lp1_,
            # sqrt_2mp1_,
            # sqrt_rat0_m_,
            # sqrt_rat3_lm__,
            # sqrt_rat4_lm__,
        )[:6]

        eta = torch.pi / k_p_r_max
        a_x_u_xxx_ = xxnufft3d3(
            n_qk,
            2 * torch.pi * k_c_0_qk_ * eta,
            2 * torch.pi * k_c_1_qk_ * eta,
            2 * torch.pi * k_c_2_qk_ * eta,
            a_k_p_qk_ * weight_3d_k_p_qk_,
            +1,
            1e-12,
            n_xxx_u,
            x_u_0.ravel() / eta,
            x_u_1.ravel() / eta,
            x_u_2.ravel() / eta,
        )

        obj = cls(
            a_x_u_xxx_, # type: ignore
            a_k_p_qk_,
            sqrt_2lp1_,
            sqrt_2mp1_,
            sqrt_rat0_m_,
            sqrt_rat3_lm__,
            sqrt_rat4_lm__,
        )

        return obj



