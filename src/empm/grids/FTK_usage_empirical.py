from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, TYPE_CHECKING
from typing_extensions import Self

if TYPE_CHECKING:
    from .PolarGrid import PolarGrid
    from empm.parameters import Parameters


# TODO: Proper import for tfh_FTK_4 fn, sig reproduced below
def tfh_FTK_4(
    parameter: Parameters,
    n_k_p_r: int,
    r8_k_p_r_: Tensor,
    r8_k_p_r_max: int
) -> dict:
    ...


# as defined by the values in the object that actually see use,
# the FTK object should consist of the following only:
# Fields used in the main loop itself:
#   r8_svd_d_max: float
#   n_delta_v: int
#   n_svd_l: int
#
# the constructed FTK is also passed into the
# tfpmh_Z_cluster_wrap_SM__14 function, which makes use of:
#   flag_tf_vs_bf: int
#   r8_delta_x_: Tensor
#   r8_delta_y_: Tensor
#
# this also passes the FTK to tfpmhp_Z_wSM___14 (3x),
# and that directly uses:
#   c16_svd_U_d_expiw_s__: Tensor
#
# and also passes the object to tpmh_VUXM_lwnM____3,
# which uses:
#   i4_svd_l_: Tensor
#   r8_svd_chebval_V_r_: Tensor
#
# or to tpmh_UXTM_dwnM____0
#   but that uses no fields not already mentioned
#
# tfpmhp also passes to one of:
#   tfpmh_UX_T_M_l2_dM__1 or
#   tfpmh_UX_T_M_l2_dM__0
# but neither of these use any fields not already mentioned.

class FTK():
    r8_svd_d_max: float
    n_delta_v: int
    n_svd_l: int
    flag_tf_vs_bf: int
    r8_delta_x_: Tensor
    r8_delta_y_: Tensor
    c16_svd_U_d_expiw_s__: Tensor
    i4_svd_l_: Tensor
    r8_svd_chebval_V_r_: Tensor


    def __init__(self,
        r8_svd_d_max: float,
        n_delta_v: int,
        n_svd_l: int,
        flag_tf_vs_bf: int,
        r8_delta_x_: Tensor,
        r8_delta_y_: Tensor,
        c16_svd_U_d_expiw_s__: Tensor,
        i4_svd_l_: Tensor,
        r8_svd_chebval_V_r_: Tensor
    ):
        self.r8_svd_d_max = r8_svd_d_max
        self.n_delta_v = n_delta_v
        self.n_svd_l = n_svd_l
        self.flag_tf_vs_bf = flag_tf_vs_bf
        self.r8_delta_x_ = r8_delta_x_
        self.r8_delta_y_ = r8_delta_y_
        self.c16_svd_U_d_expiw_s__ = c16_svd_U_d_expiw_s__
        self.i4_svd_l_ = i4_svd_l_
        self.r8_svd_chebval_V_r_ = r8_svd_chebval_V_r_


    @classmethod
    def from_grid(cls, grid: PolarGrid, params: Parameters) -> Self:
        _, blob = tfh_FTK_4(params, grid.n_k_p_r, grid.k_p_r_, grid.k_p_r_max)
        assert blob['r8_svd_d_max'] >= params.delta_r_max
        assert blob['n_delta_v'] >= params.n_delta_v_requested
        return cls(
            blob['r8_svd_d_max'],
            blob['n_delta_v'],
            blob['n_svd_l'],
            blob['flag_tf_vs_bf'],
            blob['r8_delta_x_'],
            blob['r8_delta_y_'],
            blob['c16_svd_U_d_expiw_s__'],
            blob['i4_svd_l_'],
            blob['r8_svd_chebval_V_r_'],
        )


    def to_dict(self) -> dict[str, Any]:
        res = {}
        res['type'] = 'FTK'
        res['r8_svd_d_max'] = self.r8_svd_d_max
        res['n_delta_v'] = self.n_delta_v
        res['n_svd_l'] = self.n_svd_l
        res['flag_tf_vs_bf'] = self.flag_tf_vs_bf
        res['r8_delta_x_'] = self.r8_delta_x_
        res['r8_delta_y_'] = self.r8_delta_y_
        res['c16_svd_U_d_expiw_s__'] = self.c16_svd_U_d_expiw_s__
        res['i4_svd_l_'] = self.i4_svd_l_
        res['r8_svd_chebval_V_r_'] = self.r8_svd_chebval_V_r_

        return res
