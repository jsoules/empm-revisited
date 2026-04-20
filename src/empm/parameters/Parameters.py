from __future__ import annotations

from inspect import get_annotations
from math import floor
from typing_extensions import Self
from typing import Any, TYPE_CHECKING
import torch
from torch import Tensor
from scipy.io import savemat

if TYPE_CHECKING:
    from empm.stacks import CTF, ImageStack
    from empm.grids import PolarGrid

# TODO: Even better to use a data class or something
# TODO: Better handling of random seed--need to support unset seed
# TODO: Need to get earlier access to k_p_r_max, incorporate as part of params

MACHINE_TOLERANCE = 1e-6

class Parameters():
    """Centralized collection for all configuration parameters for EMPM. Ideally
    should be largely immutable, with a few exceptions.

    Attributes:
        flag_verbose (int): Controls the verbosity of logging
        stored_verbosity (int): Internal. Allows returning to prior
            verbosity level when it is temporarily adjusted.
        n_iteration (int): Maximum iterations to run for EMPM. Defaults to 32.
        tolerance_master (float): Default tolerance for computations when more
            specific tolerances are not set. Defaults to 1e-2.
        tolerance_cluster (float): Tolerance for CTF-clustering algorithm.
            Defaults to tolerance_master.
        tolerance_pm (float): Tolerance used in computing principal modes.
            Defaults to tolerance_master.
        flag_gpu (int): Whether to attempt using GPU (defaults to 0/False)
        rseed (int | None): If set, will provide a random seed. TODO:
            Ensure this is only seeded once!
        order_limit_MS (int): ?? Defaults to -1.
        delta_r_max (float): Maximum accumulation value for pose alignment discovery.
            When updating poses, displacement values are allowed to accumulate up to
            this value without recomputing the principal-mode representation of the
            images; if the accumulated displacement exceeds this amount, the accumulated
            displacement buffer is transferred to the main displacement amount and the
            PM representations of the images are re-computed. Defaults to 0.0, in which
            case the image representations will be recomputed every pass.
        delta_r_upb (float): Maximum total allowable accumulation value for Image
            displacement. Pose estimation will not accept further displacements
            beyond this value. Defaults to twice the max per-iteration value.
        delta_r_upd_threshold (float): Threshold beyond which displacement accumulation
            will be transferred to the main displacement from the intermediate
            displacement accumulator. TODO: Better explanation. Defaults to 0.0.
        n_delta_v_requested (int): ??? Used in FTK.
        template_viewing_k_eq_d (float): Angular difference between any two
            grid points on the quadrature grid, measured at the equator. Defaults
            to 1/the highest frequency of the Fourier-space representation
            (k_p_r_max).
            TODO: NEED ACCESS TO K_P_R_MAX
        svd_eps (float): Lower bound tolerance for SVD magnitude. Defaults to
            value set for tolerance_master.
        flag_save_stage (int): Controls which operations will write log files
            (default 0)
        fname_pre (str): Prefix for log-file filenames (default to '')
        flag_alternate_MS_vs_SM (int): Controls the pattern of image-template
            best-fit assignments. During EMPM, on each cycle, a match must be
            made so that each image is assigned to one template. This can be
            done by assigning the image to the template that best matches it,
            but with unfavorable initial conditions, this may result in some templates
            being starved (with no assigned image) while others are over-represented
            (with many images assigned). Alternatively, one can assign each template
            the image that fits it best (even if that image would be a better fit for
            other templates), ensuring that there is data to adjust every template.
            If this flag is set nonzero, the EMPM algorithm will alternate between
            image-first assignments and template-first assignments on every iteration.
            If it is set to 0, the algorithm will do image-first alignment for the
            first half of its iterations and template-first alignment for the latter
            half. Defaults to 1 (alternating each iteration).

        sample_sphere_k_eq_d (float): A copy of k_eq_d from spherical-grid creation
        flag_qbp_vs_lsq (int): Whether volumetric reconstruction should use quadrature
            backpropagation (1) or least squares (0)
        qbp_eps (float): Threshold for the local pseudo-inverse used in quadrature
            backpropgataion. Unused for other volumetric reconstruction methods.
            Defaults to master tolerance if not set.
        n_x_u_pack (int): Packing for ?? TODO

        r8_delta_r_max (float): Maximum displcement of individual images used in FTK.
            Defaults to value set for delta_r_max. (FTK)
        r8_svd_eps (float): Lower bound tolerance for SVD magnitude used in FTK.
            Defaults to value set for svd_eps. (FTK)
        r8_delta_x_requested_ (Tensor): Array of (per-image?) x-displacements
            requested for FTK. (FTK)
        r8_delta_y_requested_ (Tensor): Array of (per-image?) y-displacements
            requested for FTK. (FTK)
        l_max (int): ?? for FTK. Defaults to 25. (FTK)
        n_a_degree (int): ?? for FTK. Defaults to 64. (FTK)
        n_b_degree (int): ?? for FTK. Defaults to 65. (FTK)
        flag_p_vs_c (int): ?? for FTK. Defaults to 0. (FTK)
        flag_tf_vs_bf (int): ?? for FTK. Defaults to 1. (FTK)

        flag_precompute_M_k_q_wkM__ (int): Internal. If nonzero, precompute
            ??. Currently hard-coded to 1.
        flag_precompute_UX_T_M_l2_dM__ (int): Internal. If nonzero, precompute
            ??. Currently hard-coded to 1.
        flag_precompute_UX_M_l2_M_ (int): Internal. If nonzero, precompute
            ??. Currently hard-coded to 1.
        flag_precompute_svd_V_UX_M_lwnM____ (int): Internal. If nonzero, precompute
            ??. Currently hard-coded to 1.
        flag_precompute_UX_CTF_S_k_q_wnS__ (int): Internal. If nonzero, precompute
            ??. Currently hard-coded to 1.
        flag_precompute_UX_CTF_S_l2_S_ (int): Internal. If nonzero, precompute
            ??. Currently hard-coded to 1.

        flag_rank_vs_tolerance (int): Controls clustering behavior in clustering
            CTFs (?)  Defaults to 0.
        flag_clump_vs_cluster (int): Controls behavior in computing CTF clusters
            (?). Defaults to the value of flag_rank_vs_tolerance.
        rank_pm (int): Rank/number of principal modes to use for principal-modes
            representation of translation matrix (?). Defaults to 10.
        rank_CTF (int): Expected rank of overall CTF matrix, which affects
            CTF clustering. If negative (the default), will attempt to determine
            empirically through SVD.
    """

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
    delta_r_upd_threshold: float
    n_delta_v_requested: int
    template_viewing_k_eq_d: float
    svd_eps: float
    flag_save_stage: int
    fname_pre: str
    flag_alternate_MS_vs_SM: int

    sample_sphere_k_eq_d: float
    flag_qbp_vs_lsq: int
    qbp_eps: float
    n_x_u_pack: int

    # FTK-related
    r8_delta_r_max: float
    r8_svd_eps: float
    r8_delta_x_requested_: Tensor
    r8_delta_y_requested_: Tensor
    l_max: int
    n_a_degree: int
    n_b_degree: int
    flag_p_vs_c: int
    flag_tf_vs_bf: int

    # Why aren't these bools?
    flag_precompute_M_k_q_wkM__: int
    flag_precompute_UX_T_M_l2_dM__: int
    flag_precompute_UX_M_l2_M_: int
    flag_precompute_svd_V_UX_M_lwnM____: int
    flag_precompute_UX_CTF_S_k_q_wnS__: int
    flag_precompute_UX_CTF_S_l2_S_: int

    # From tfpmut_wrap_6.py, aka coordinating multiple runs of empm loop
    flag_rank_vs_tolerance: int
    flag_clump_vs_cluster: int
    rank_pm: int
    rank_CTF: int


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
        delta_r_upd_threshold: float = 0.0,
        n_delta_v_requested: int = 0,
        template_viewing_k_eq_d: float = -1.,
        svd_eps: float = -1.,
        flag_save_stage: int = 0,
        fname_pre: str = '',
        flag_alternate_MS_vs_SM: int = 1,
        sample_sphere_k_eq_d: float = 0.0,  # TODO: This describes a grid and maybe shouldn't be here?
        flag_qbp_vs_lsq: int = 1,
        qbp_eps: float = -1.,
        n_x_u_pack: int = -1,
        r8_delta_r_max: float = -1.,
        r8_svd_eps: float = -1.,
        r8_delta_x_requested_: Tensor = torch.zeros(0),
        r8_delta_y_requested_: Tensor = torch.zeros(0),
        l_max: int = 25,
        n_a_degree: int = 64,
        n_b_degree: int = 65,
        flag_p_vs_c: int = 0,
        flag_tf_vs_bf: int = 1,
        flag_rank_vs_tolerance: int = 0,
        flag_clump_vs_cluster: int = -1,
        rank_pm: int = 10,
        rank_CTF: int = -1,
        k_p_r_max: float = 0,   # TODO improve this
    ):
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
        self.tolerance_pm = tolerance_pm
        if tolerance_cluster < 0:
            self.tolerance_cluster = self.tolerance_master
        if tolerance_pm < 0:
            self.tolerance_pm = self.tolerance_master
        self.flag_gpu = flag_gpu
        self.rseed = rseed
        self.order_limit_MS = order_limit_MS
        self.delta_r_max = delta_r_max
        self.delta_r_upb = delta_r_upb
        self.delta_r_upd_threshold = delta_r_upd_threshold
        self.n_delta_v_requested = n_delta_v_requested
        self.template_viewing_k_eq_d = template_viewing_k_eq_d
        self.svd_eps = svd_eps if svd_eps > 0 else tolerance_master
        self.flag_save_stage = flag_save_stage
        self.fname_pre = fname_pre
        self.flag_alternate_MS_vs_SM = flag_alternate_MS_vs_SM

        self.sample_sphere_k_eq_d = sample_sphere_k_eq_d
        self.flag_qbp_vs_lsq = flag_qbp_vs_lsq
        self.qbp_eps = qbp_eps if qbp_eps > 0 else self.tolerance_master
        self.n_x_u_pack = n_x_u_pack

        # FTK-related
        self.r8_delta_r_max = r8_delta_r_max if r8_delta_r_max > 0 else self.delta_r_max
        self.r8_svd_eps = r8_svd_eps if r8_svd_eps > 0 else self.svd_eps
        # These 2 need to go back to None if they are length 0
        self.r8_delta_x_requested_ = r8_delta_x_requested_
        self.r8_delta_y_requested_ = r8_delta_y_requested_
        self.l_max = l_max
        self.n_a_degree = n_a_degree
        self.n_b_degree = n_b_degree
        self.flag_p_vs_c = flag_p_vs_c
        self.flag_tf_vs_bf = flag_tf_vs_bf

        # empm_loop wrapper level
        self.flag_rank_vs_tolerance = flag_rank_vs_tolerance
        if flag_clump_vs_cluster < 0:
            # if unset, should be whatever the rank_vs_tolerance flag was
            flag_clump_vs_cluster = flag_rank_vs_tolerance
        self.flag_clump_vs_cluster = flag_clump_vs_cluster
        self.rank_pm = rank_pm
        self.rank_CTF = rank_CTF

        self.flag_precompute_M_k_q_wkM__ = 1
        self.flag_precompute_UX_T_M_l2_dM__ = 1
        self.flag_precompute_UX_M_l2_M_ = 1
        self.flag_precompute_svd_V_UX_M_lwnM____ = 1
        self.flag_precompute_UX_CTF_S_k_q_wnS__ = 1
        self.flag_precompute_UX_CTF_S_l2_S_ = 1


    def check_equality(self, other: Self) -> bool:
        if not isinstance(other, type(self)):
            return False

        # NOTE: This relies on all members being type-annotated
        annots = get_annotations(self)
        for k in annots.keys():
            try:
                mine = getattr(self, k)
                theirs = getattr(other, k)
                if annots[k] == 'Tensor':
                    if not torch.all(torch.isclose(mine, theirs)).item():
                        return False
                else:
                    if mine != theirs:
                        return False
            except:
                return False

        return True


    def get_ms_vs_sm(self, n_iteration: int = 0) -> bool:
        if self.flag_alternate_MS_vs_SM != 0:
            return n_iteration % 2 == 0
        return n_iteration < floor(self.n_iteration / 2)


    def set_empirical_ctf_rank(self, ctf: CTF, grid: PolarGrid, imgs: ImageStack):
        if self.rank_CTF > 0:
            raise Exception("Attempt to set empirical CTF rank when a non-negative one was manually set")
        rank = ctf.empirically_determine_rank(self, grid, imgs.n_M)
        self.rank_CTF = rank


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


    def to_dict(self, n_iter: int = 0) -> dict:
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
        res['delta_r_upd_threshold'] = self.delta_r_upd_threshold
        res['n_delta_v_requested'] = self.n_delta_v_requested
        res['template_viewing_k_eq_d'] = self.template_viewing_k_eq_d
        res['svd_eps'] = self.svd_eps
        res['flag_save_stage'] = self.flag_save_stage
        res['fname_pre'] = self.fname_pre
        res['flag_alternate_MS_vs_SM'] = self.flag_alternate_MS_vs_SM
        res['flag_MS_vs_SM'] = 1 if self.get_ms_vs_sm(n_iter) else 0

        res['sample_sphere_k_eq_d'] = self.sample_sphere_k_eq_d
        res['flag_qbp_vs_lsq'] = self.flag_qbp_vs_lsq
        res['qbp_eps'] = self.qbp_eps
        res['n_x_u_pack'] = self.n_x_u_pack

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

        # From tfpmut_wrap_6 / coordinating multiple empm-loop runs
        res['flag_rank_vs_tolerance'] = self.flag_rank_vs_tolerance
        res['flag_clump_vs_cluster'] = self.flag_clump_vs_cluster
        res['rank_pm'] = self.rank_pm
        res['rank_CTF'] = self.rank_CTF

        return res


    @classmethod
    def from_dict(cls, d: dict) -> Self:
        d_type = d.get('type', 'some_default')
        if d_type != 'parameter':
            raise ValueError("Non-parameter dictionary not valid for creating Parameters object.")
        del d['type']
        del d['flag_MS_vs_SM']

        # delete the precompute flags as they're all hard-coded right now
        hard_coded = [
            "flag_precompute_M_k_q_wkM__",
            "flag_precompute_UX_T_M_l2_dM__",
            "flag_precompute_UX_M_l2_M_",
            "flag_precompute_svd_V_UX_M_lwnM____",
            "flag_precompute_UX_CTF_S_k_q_wnS__",
            "flag_precompute_UX_CTF_S_l2_S_",
        ]
        for x in hard_coded:
            del d[x]
        if d['rseed'] == 0:
            del d['rseed']

        res = cls(**d)
        return res


def matlab_save(fname: str, data: dict[str, Any]) -> None:
    for key in data:
        # Before saving, for all tensors, reorder dimensions to match matlab
        if isinstance(data[key], torch.Tensor):
            t = data[key]
            data[key] = torch.permute(t, list(reversed(t.shape)))
    savemat(file_name=fname, mdict=data, oned_as='column')
