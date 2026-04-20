import torch
from sys import argv

from dir_empm.dir_matlab_macros import matlab_load
from empm import Parameters, PolarGrid, CTF, ImageStack, Volume, Poses, empm_loop_wrapper



def _set_params() -> Parameters:
    p = Parameters(
        flag_verbose = 1,
        n_iteration = 32,
        rseed = 0,
        tolerance_master = 0.01,
        tolerance_pm = 0.1,
        flag_gpu = 1,
        delta_r_max = 0.017631663493955,
        delta_r_upb = 0.141053307951639,
        # dir_tfpm??
        flag_clump_vs_cluster = 0,
        rank_CTF = 4,
        k_p_r_max = 7.639437268410976,
        fname_pre = 'tmp',      # NOTE: Orig has tmp_dir_tfpm_mat/test_tfpmu_wrap_6_X[A|B]_from_python
                                # but a) no hard-coding paths and b) can't really write to Adi's dir anyway
        sample_sphere_k_eq_d = 0.159154943091895,
        flag_qbp_vs_lsq = 1,
        qbp_eps = 0.01,
        # flag_force_create_mat = 0
        # flag_force_create_tmp = 1
        # parameter['half_diameter_x_c']=1;
        # parameter['n_x_u_pack']=64;
        # parameter['cg_lsq_n_order']=5;
        # parameter['date_diff_threshold']=0.250000000000000;
        # parameter['str_strategy_prefix']='';
        # parameter['n_complete_calculation']=0;

        # these match the defaults but I'm including them explicitly
        # in case the defaults change
        flag_rank_vs_tolerance = 0,
        flag_alternate_MS_vs_SM = 1,
        rank_pm = 10,
    )
    return p


def _set_grid(matlab_src: dict) -> PolarGrid:
    g = PolarGrid.make_uniform_grid(
        n_k_p_r = int(matlab_src['n_k_p_r'].item()),
        k_p_r_ = matlab_src['k_p_r_'].to(dtype=torch.float32),
        k_p_r_max = float(matlab_src['k_p_r_max'].item()),
        n_w_0in_ = matlab_src['n_w_'].to(dtype=torch.int32),
        # TODO: template_k_eq_d
        weight_3d_k_p_r_ = matlab_src['weight_3d_k_p_r_'].to(dtype=torch.float32)
    )
    return g


def _set_ctfs(matlab_src: dict) -> CTF:
    ctfs = CTF(
        # n_CTF = int(matlab_src['n_CTF'].item()), # we don't actually need this--just read it from the actual CTF tensor
        CTF_k_p_wkC__ = matlab_src['CTF_k_p_wkC__'].to(dtype=torch.float32),
        index_nCTF_from_nM_ = matlab_src['index_nCTF_from_nM_'].to(dtype=torch.int32)
    )
    return ctfs


def _set_poses(matlab_src: dict) -> Poses:
    p = Poses(
        euler_polar_a_M_ = matlab_src['tmp_XA_euler_polar_a_M_'].to(dtype=torch.float32),
        euler_azimu_b_M_ = matlab_src['tmp_XA_euler_azimu_b_M_'].to(dtype=torch.float32),
        euler_gamma_z_M_ = matlab_src['tmp_XA_euler_gamma_z_M_'].to(dtype=torch.float32),
        image_delta_x_acc_M_= matlab_src['tmp_XA_image_delta_x_M_'].to(dtype=torch.float32),
        image_delta_y_acc_M_= matlab_src['tmp_XA_image_delta_y_M_'].to(dtype=torch.float32)
    )
    return p


def main(tmp_dir_base: str):
    tmp_dir_tfpm_mat = f'{tmp_dir_base}/dir_trpv1_x0/dir_tfpm_mat'
    tmp_fname_mat = f'{tmp_dir_tfpm_mat}/test_tfpmut_wrap_6_a1t0014p15r1.mat'

    _tmp = matlab_load(fname_mat = tmp_fname_mat)
    parameter = _set_params()
    grid = _set_grid(_tmp)
    ctfs = _set_ctfs(_tmp)
    weight_3d_k_p_r_ = _tmp['weight_3d_k_p_r_'].to(dtype=torch.float32)
    images = ImageStack(_tmp['M_k_p_wkM__'].to(dtype=torch.complex64))
    volume = Volume(l_max_ = _tmp['l_max_'].to(dtype=torch.int32))

    parameter.fname_pre = 'tmp/A/'
    empm_loop_wrapper(
        parameter,
        grid,
        weight_3d_k_p_r_,
        ctfs,
        images,
        volume,
    )

    parameter.fname_pre = 'tmp/B/'
    volume = Volume(
        l_max_ = _tmp['l_max_'].to(dtype=torch.int32),
        a_k_Y_reco_yk_= _tmp['tmp_XA_a_k_Y_reco_yk_'].to(dtype=torch.complex64)
    )
    initial_poses = _set_poses(_tmp)

    empm_loop_wrapper(
        parameter,
        grid,
        weight_3d_k_p_r_,
        ctfs,
        images,
        volume,
        initial_poses
    )


# TODO: The below entries should be configured in an incoming Poses object
# euler_polar_a_ini_M_=None,
# euler_azimu_b_ini_M_=None,
# euler_gamma_z_ini_M_=None,
# image_delta_x_acc_ini_M_=None, --> maps to acc
# image_delta_y_acc_ini_M_=None,

# def empm_loop_wrapper(  # former tfpmut_wrap_6. Name to be further revised.
#         parameter: Parameters,
#         grid: PolarGrid,
#         weight_3d_k_p_r_: Tensor,
#         ctfs: CTF,
#         images: ImageStack,
#         volume: Volume,
#         initial_poses: Poses | None = None,
#         delta_sigma_base: float = 0.0, # TODO: could this be vector-valued?
#         ftk: FTK | None = None
# ):


if __name__ == '__main__':
    if argv[1] is None:
        raise ValueError("Must specify path to test_tfpmut_wrap_6_a1t0014p15r1.mat")

    main(argv[1])

