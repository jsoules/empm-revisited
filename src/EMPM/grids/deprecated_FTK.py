import torch
from torch import Tensor

class FTK():

    r8_svd_d_max: float
    n_delta_v: int
    n_svd_l: int
    r8_delta_r_max: float
    flag_p_vs_c: int
    flag_tf_vs_bf: int

    # midway through tfh_FTK_4
    n_delta_v: int
    r8_delta_x_: Tensor
    r8_delta_y_: Tensor

    # from gen_Jsvd_FTK_8
    n_r_degree: int         # defaulted to 32
    n_d_degree: int         # defaulted to 32
    # l_max: int              # defaulted to 24
    n_svd_r: int
    r8_svd_r_: Tensor
    r8_svd_r_m: float
    r8_svd_r_c: float       # identical to r8_svd_r_m
    n_svd_d: int
    r8_svd_d_: Tensor
    i4_svd_l_: Tensor
    r8_svd_r_: Tensor
    r8_svd_s_: Tensor
    r8_svd_U_d_chebcoef_: Tensor
    r8_svd_V_r_chebcoef_: Tensor


    # at end of tfh_FTK_4
    r8_svd_chebval_U_d_: Tensor
    r8_svd_r_max: float
    r8_svd_chebval_V_r_: Tensor

    c16_svd_expiw__: Tensor
    c16_svd_expiw_s__: Tensor




    def __init__(self):
        ...

    
    def to_dict(self):
        res = {
            'type': 'FTK'
        }


