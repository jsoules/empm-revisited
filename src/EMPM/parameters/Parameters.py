from typing_extensions import Self
from typing import Any
import torch
from scipy.io import savemat

# TODO: Even better to use a data class or something
# TODO: Better handling of random seed--need to support unset seed
# TODO: Need to get earlier access to k_p_r_max, incorporate as part of params

class Parameters():
    flag_verbose: int
    stored_verbosity: int
    n_iteration: int
    tolerance_master: float
    tolerance_cluster: float
    tolerance_pm: float
    flag_gpu: int
    rseed: int | None
    order_limit_MS: int
    delta_r_max: float
    delta_r_upb: float
    n_delta_v_requested: int
    template_viewing_k_eq_d: float
    flag_save_stage: int
    fname_pre: str
    flag_alternate_MS_vs_SM: int
    delta_r_upd_threshold: float

    # FTK-related
    r8_delta_r_max: float                   # default whatever delta r max was, or else 0.0
    r8_svd_eps: float                       # default to whatever svd_eps is
    r8_delta_x_requested_: torch.Tensor     # default None
    r8_delta_y_requested_: torch.Tensor     # default None
    l_max: int                              # default 25
    n_a_degree: int                         # default 64
    n_b_degree: int                         # default 65 why not
    flag_p_vs_c: int                        # default 0
    flag_tf_vs_bf: int                      # default 1

    # Why aren't these bools?
    flag_precompute_M_k_q_wkM__: int
    flag_precompute_UX_T_M_l2_dM__: int
    flag_precompute_UX_M_l2_M_: int
    flag_precompute_svd_V_UX_M_lwnM____: int
    flag_precompute_UX_CTF_S_k_q_wnS__: int
    flag_precompute_UX_CTF_S_l2_S_: int


    def __init__(self, *,
        flag_verbose: int = 0,
        n_iteration: int = 32,
        tolerance_master: float = 1e-2,
        tolerance_cluster: float = -1.,
        tolerance_pm: float = -1.,
        flag_gpu: int = 0,
        rseed: int | None = None,
        order_limit_MS: int = -1,
        delta_r_max: float = 0.1,
        delta_r_upb: float = -1.,
        n_delta_v_requested: int = 0,
        template_viewing_k_eq_d: float = -1.,
        flag_save_stage: int = 0,
        fname_pre: str = '',
        flag_alternate_MS_vs_SM: int = 1,
        delta_r_upd_threshold: float = 0.0,
        r8_delta_r_max: float = 0.0,
        r8_svd_eps: float = 1e-4,
        r8_delta_x_requested_: torch.Tensor = torch.zeros(0),
        r8_delta_y_requested_: torch.Tensor = torch.zeros(0),
        l_max: int = 25,
        n_a_degree: int = 64,
        n_b_degree: int = 65,
        flag_p_vs_c: int = 0,
        flag_tf_vs_bf: int = 1,
    ):
        k_p_r_max = 0.
        if delta_r_upb < 0:
            delta_r_upb = 2 * delta_r_max
        if template_viewing_k_eq_d < 0:
            # TODO: get access to k_p_r_max
            template_viewing_k_eq_d = 1.0/max(1e-12, k_p_r_max)
        
        self.flag_verbose = flag_verbose
        self.stored_verbosity = flag_verbose
        self.n_iteration = n_iteration
        self.tolerance_master = tolerance_master
        self.tolerance_cluster = tolerance_cluster
        if tolerance_cluster < 0:
            self.tolerance_cluster = self.tolerance_master
        if tolerance_pm < 0:
            self.tolerance_pm = self.tolerance_master
        self.flag_gpu = flag_gpu
        self.rseed = rseed
        self.order_limit_MS = order_limit_MS
        self.delta_r_max = delta_r_max
        self.delta_r_upb = delta_r_upb
        self.n_delta_v_requested = n_delta_v_requested
        self.template_viewing_k_eq_d = template_viewing_k_eq_d
        self.flag_save_stage = flag_save_stage
        self.fname_pre = fname_pre
        self.flag_alternate_MS_vs_SM = flag_alternate_MS_vs_SM
        self.delta_r_upd_threshold = delta_r_upd_threshold

        # FTK-related
        self.r8_delta_r_max = r8_delta_r_max
        self.r8_svd_eps = r8_svd_eps
        # These 2 need to go back to None if they are length 0
        self.r8_delta_x_requested_ = r8_delta_x_requested_
        self.r8_delta_y_requested_ = r8_delta_y_requested_
        self.l_max = l_max
        self.n_a_degree = n_a_degree
        self.n_b_degree = n_b_degree
        self.flag_p_vs_c = flag_p_vs_c
        self.flag_tf_vs_bf = flag_tf_vs_bf

        self.flag_precompute_M_k_q_wkM__ = 1
        self.flag_precompute_UX_T_M_l2_dM__ = 1
        self.flag_precompute_UX_M_l2_M_ = 1
        self.flag_precompute_svd_V_UX_M_lwnM____ = 1
        self.flag_precompute_UX_CTF_S_k_q_wnS__ = 1
        self.flag_precompute_UX_CTF_S_l2_S_ = 1


    def write_to_file(self):
        raise NotImplementedError
    

    @classmethod
    def read_from_file(cls, fn: str) -> Self:
        raise NotImplementedError
    

    def print_per_verbosity(self, string: str, min_verbosity: int = 1) -> None:
        if self.flag_verbose < min_verbosity: return
        print(string)


    def reduce_verbosity(self):
        self.stored_verbosity = self.flag_verbose
        self.flag_verbose = max(0, self.flag_verbose - 1)


    def restore_verbosity(self):
        self.flag_verbose = self.stored_verbosity


    # TODO: Support a flag for matlab vs normal ordering
    def save_report(self, name_tail: str, data: dict[str, Any], min_level: int = 3) -> None:
        if self.flag_save_stage < min_level: return
        fname = f"{self.fname_pre}{name_tail}"
        print(f" %% writing {fname}")
        matlab_save(fname, data)


    def to_dict(self) -> dict:
        res = {}
        res['type'] = 'parameter'
        res['flag_verbose'] = self.flag_verbose
        res['n_iteration'] = self.n_iteration
        res['tolerance_master'] = self.tolerance_master
        res['tolerance_cluster'] = self.tolerance_cluster
        res['tolerance_pm'] = self.tolerance_pm
        res['flag_gpu'] = self.flag_gpu
        res['rseed'] = 0 if self.rseed is None else self.rseed # NOTE: THIS FORCES MANUAL SEEDING OF RNG
        res['order_limit_MS'] = self.order_limit_MS
        res['delta_r_max'] = self.delta_r_max
        res['delta_r_upb'] = self.delta_r_upb
        res['n_delta_v_requested'] = self.n_delta_v_requested
        res['template_viewing_k_eq_d'] = self.template_viewing_k_eq_d
        res['flag_save_stage'] = self.flag_save_stage
        res['fname_pre'] = self.fname_pre
        res['flag_alternate_MS_vs_SM'] = self.flag_alternate_MS_vs_SM
        res['flag_MS_vs_SM'] = 1

        res['flag_precompute_M_k_q_wkM__'] = self.flag_precompute_M_k_q_wkM__
        res['flag_precompute_UX_T_M_l2_dM__'] = self.flag_precompute_UX_T_M_l2_dM__
        res['flag_precompute_UX_M_l2_M_'] = self.flag_precompute_UX_M_l2_M_
        res['flag_precompute_svd_V_UX_M_lwnM____'] = self.flag_precompute_svd_V_UX_M_lwnM____
        res['flag_precompute_UX_CTF_S_k_q_wnS__'] = self.flag_precompute_UX_CTF_S_k_q_wnS__
        res['flag_precompute_UX_CTF_S_l2_S_'] = self.flag_precompute_UX_CTF_S_l2_S_

        # FTK-related
        res['r8_delta_r_max'] = self.r8_delta_r_max
        res['r8_svd_eps'] = self.r8_svd_eps
        res['r8_delta_x_requested_'] = self.r8_delta_x_requested_ if self.r8_delta_x_requested_.numel() > 0 else None
        res['r8_delta_y_requested_'] = self.r8_delta_y_requested_ if self.r8_delta_y_requested_.numel() > 0 else None
        res['l_max'] = self.l_max
        res['n_a_degree'] = self.n_a_degree
        res['n_b_degree'] = self.n_b_degree
        res['flag_p_vs_c'] = self.flag_p_vs_c
        res['flag_tf_vs_bf'] = self.flag_tf_vs_bf

        return res


def matlab_save(fname: str, data: dict[str, Any]) -> None:
    for key in data:
        # Before saving, for all tensors, reorder dimensions to match matlab
        if isinstance(data[key], torch.Tensor):
            t = data[key]
            data[key] = torch.permute(t, list(reversed(t.shape)))
    savemat(file_name=fname, mdict=data, oned_as='column')
