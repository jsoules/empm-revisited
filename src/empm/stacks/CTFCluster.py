from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

from dir_empm.knn_cluster_CTF_k_p_r_kC__1 import knn_cluster_CTF_k_p_r_kC__1
from dir_empm.principled_marching_empirical_cost_matrix_2 import principled_marching_empirical_cost_matrix_2
from dir_empm.principled_marching_cost_matrix_7 import principled_marching_cost_matrix_7

from empm.parameters import MACHINE_TOLERANCE
from empm.util import matlab_style_svd_macro

if TYPE_CHECKING:
    from . import CTF, ImageStack, Volume
    from empm.grids import PolarGrid
    from empm.parameters import Parameters

## FYI
# # # def knn_cluster_CTF_k_p_r_kC__1(
# # #     parameter=None, # parameter hive
# # #     n_k_p_r=None,   # grid property
# # #     k_p_r_=None,    # grid property
# # #     weight_2d_k_p_r_=None,  # grid property
# # #     n_CTF=None,     # CTF property
# # #     CTF_k_p_r_kC__=None,    # alteration of CTF property
# # # ) -> tuple[Parameters, Tensor]:
# # #     ...

# # # def principled_marching_empirical_cost_matrix_2(
# # #         n_k_p_r=None,           # grid property
# # #         k_p_r_=None,            # grid property
# # #         weight_2d_k_p_r_=None,  # grid property
# # #         n_w_=None,              # grid property
# # #         n_M=None,               # image stack property
# # #         M_k_p_wkM__=None,       # image stack
# # # ) -> tuple[Tensor, Tensor]:
# # #     ...

# # # def principled_marching_cost_matrix_7(
# # #         n_k_p_r=None,           # grid property
# # #         k_p_r_=None,            # grid property
# # #         weight_k_p_r_=None,     # grid property
# # #         l_max_=None,            # spherical harmonic property
# # #         n_molecule=None,        # NOT USED FOR THIS CALL PATH
# # #         molecule_density_=None, # NOT USED FOR THIS CALL PATH
# # #         a_k_Y_ykv__=None,       # the empirical cost matrix I guess
# # #         CTF_k_p_r_xcor_kk__=None, # per-cluster averaged something or other
# # #         delta_sigma=None,       # base translation; float
# # #         pm_delta_integral_tolerance=None, # not used by us
# # # ) -> tuple[Tensor, Tensor, Tensor, Tensor, float, int, Tensor]:
# # #         # NOTE: for our purposes we only care about the first 2 returns
# # #         # X_kk__,
# # #         # X_weight_r_,
# # #         # X_ori_kk__,
# # #         # X_tau_kk__,
# # #         # weight_so3,
# # #         # n_m_max,
# # #         # polar_a_,
# # #     ...


def force_isotropy(ctfs: CTF, grid: PolarGrid) -> Tensor:
    """Takes a stack of CTFs expressed as one value per grid point and returns
    a stack with a single weight per ring, that weight being the average over
    all the inplane-angle coordinates. This is elsewhere referred to
    as CTF_k_p_r_kC__ and is indexed by the CTF index, then by the radial index.

    Args:
        ctfs (CTF): Object holding stack of CTFs to average into isotropicity
        grid (PolarGrid): Polar grid fitted by the CTFs

    Returns:
        Tensor: CTF_k_p_r_kC__, a tensor indexed by CTF, then by radial coordinate,
        representing an isotropic CTF (i.e. one with the same value for every
        inplane rotation for a given radial distance)
    """
    if not grid.is_uniform:
        raise Exception("Averaging anisotropic CTFs over inplane angles to isotropize will not work for varying inplane angle counts.")
    return _force_isotropy(ctfs.CTF_k_p_wkC__, grid)
    # shape_ctf_by_radius_by_inplanes = (ctfs.n_CTF, grid.n_k_p_r, grid.n_w_max)
    # unflattened_ctfs = torch.reshape(ctfs.CTF_k_p_wkC__, shape_ctf_by_radius_by_inplanes)
    # # NOTE: taking the mean over dimension 2 will already accomplish reshaping to the
    # # shape of (former dimension 0, former dimension 1)
    # CTF_k_p_r_kC__ = torch.mean(unflattened_ctfs, dim=2)
    # return CTF_k_p_r_kC__


def _force_isotropy(ctfs: Tensor, grid: PolarGrid) -> Tensor:
    if not grid.is_uniform:
        raise Exception("Currently unsupported for nonuniform inplane angle counts")
    grid_shape = (-1, grid.n_k_p_r, grid.n_w_max)
    return ctfs.reshape(grid_shape).mean(2)


class CTFCluster():
    """Class representing clusters of CTFs, along with some
    precomputed summary properties and bookkeeping.

    Attributes:
        n_cluster (int): Number of CTF clusters
        ctfs (CTF): The set of CTFs being clustered.
            For consistency's sake, code which consumes
            the CTF clusters should access the CTFs solely
            through the cluster object, to avoid potentially
            referring to a different set of CTFs.
        index_ncluster_from_nCTF_ (Tensor): Vector mapping
            CTF index to the index of the cluster that CTF
            belongs to. Should probably be identical to
            index_ncluster_from_nM_.
        index_ncluster_from_nM_ (Tensor): Vector, indexed
            by image index, mapping to the cluster to
            which that image's CTF belongs.
        index_nM_from_ncluster__ (list[Tensor]): List,
            indexed by cluster ID, of the image indices
            which map into that cluster
        n_index_nM_from_ncluster_ (Tensor): Vector, indexed
            by cluster ID, that counts the number of images
            (and thus CTFs) that map into that cluster. Each
            entry should be the length of the corresponding
            list in index_nM_from_ncluster__.
        pm_n_UX_rank_c_ (Tensor): Per-cluster vector of the
            number of ranks we care about for PM
            compression/decompression. (The pm_UX_knc___ tensor
            needs to accommodate the highest required count, but
            we often won't use the same rank for every cluster.)
        pm_UX_knc___ (Tensor): Per-cluster weight values used
            to compress/uncompress via principal modes
        pm_X_weight_rc__ (Tensor): Values for principled marching
            reconstruction 
        CTF_k_p_r_xavg_kc__ (Tensor): A "representative isotropic"
            CTF for each cluster, indexed as cluster-index x
            radial-coordinate (isotropic, so one weight per ring)
    """

    n_cluster: int
    index_ncluster_from_nCTF_: Tensor
    ctfs: CTF

    # computed -- cluster to image index maps
    index_ncluster_from_nM_: Tensor
    index_nM_from_ncluster__: list[Tensor]
    n_index_nM_from_ncluster_: Tensor
    # computed -- principled marching
    pm_n_UX_rank_c_: Tensor
    pm_UX_knc___: Tensor
    pm_X_weight_rc__: Tensor

    # per-cluster average isotropic CTF (as a "representative")
    # indexed as cluster-index x radial-coordinate
    CTF_k_p_r_xavg_kc__: Tensor

    # TODO: provide option to use different clustering algorithm?
    def __init__(self,
        parameter: Parameters,
        grid: PolarGrid,
        ctfs: CTF,
    ):
        self.ctfs = ctfs
        CTF_k_p_r_kC__ = force_isotropy(ctfs, grid)
        (_, index_ncluster_from_nCTF_) = knn_cluster_CTF_k_p_r_kC__1(
            parameter = parameter.to_dict(),
            n_k_p_r = grid.n_k_p_r,
            k_p_r_ = grid.k_p_r_,
            weight_2d_k_p_r_ = grid.weight_2d_k_p_r_,
            n_CTF = ctfs.n_CTF,
            CTF_k_p_r_kC__ = CTF_k_p_r_kC__
        )
        
        self.index_ncluster_from_nCTF_ = index_ncluster_from_nCTF_
        self.n_cluster = 1 + int(torch.max(self.index_ncluster_from_nCTF_).item())
        self.index_ncluster_from_nM_ = \
            self.index_ncluster_from_nCTF_[ctfs.index_nCTF_from_nM_]
        self.index_nM_from_ncluster__ = []
        self.n_index_nM_from_ncluster_ = torch.zeros(self.n_cluster, dtype=torch.int32)
        # For each cluster, build a map of a) the image indices in that cluster,
        # and b) the count of images indexed into each cluster.
        for ncluster in range(self.n_cluster):
            img_indices = torch.where(self.index_ncluster_from_nM_ == ncluster)[0]
            self.index_nM_from_ncluster__.append(img_indices)
            self.n_index_nM_from_ncluster_[ncluster] = int(img_indices.numel())
        # allocate space for principal-mode matrices
        self.pm_n_UX_rank_c_ = torch.zeros((self.n_cluster), dtype=torch.int32)
        self.pm_UX_knc___ = torch.zeros((self.n_cluster, grid.n_k_p_r - 1), dtype=torch.float32)
        self.make_cluster_averages(grid)
        # don't preallocate, let's just do it since we have the means to do so
        # self.CTF_k_p_r_xavg_kc__ = torch.zeros((self.n_cluster, grid.n_k_p_r))


    # This part picks the most informative collection of linear combinations
    # this corresponds to taking the SVD of the xcorr matrix produced by either the
    # empirical or ansatz branch
    # the UX is like the recipe for compressing and decompressing
    # (it's orthonormal so UX is the compression and UX.T is the decompression)
    # (note UX is NOT square, that's why we achieve a dimension reduction)
    def determine_principal_modes(self,
        # pm_X_kkc___: Tensor,
        parameters: Parameters,
        grid: PolarGrid,
        images: ImageStack,
        volume: Volume,
        delta_sigma_base: float = 0.0   # TODO check if could be vector-valued
    ) -> None:
        if volume.a_k_Y_reco_yk_.numel() == 0:
            pm_X_kkc___ = self._determine_principal_modes_empirically(grid, images)
        else:
            pm_X_kkc___ = self._determine_principal_modes_from_ansatz(grid, volume, delta_sigma_base)
        
        # Target rank for the weight matrix is one less than the number of frequencies in the grid.
        n_UX_rank = grid.n_k_p_r - 1
        for ncluster in range(self.n_cluster):
            # # tmp_X_kk__ = torch.reshape(pm_X_kkc___[ncluster,:,:], grid_shape)
            # This is already nkpr x nkpr by construction
            tmp_X_kk__ = pm_X_kkc___[ncluster] # no need to include indexing when all dims are full
            tmp_UX__, tmp_SX_, _ = matlab_style_svd_macro(tmp_X_kk__, n_UX_rank)
            # record the pm rank for this to be the count of the SVDs greater than tolerance
            norm_val = max(MACHINE_TOLERANCE, tmp_SX_[0]) # linalg.svd returns S in desc order
            self.pm_n_UX_rank_c_[ncluster] = (tmp_SX_ / norm_val > parameters.tolerance_pm).sum().item()
            # # # significant_modes = torch.where(tmp_SX_ / normalization_val > parameters.tolerance_pm)[0]
            # # # pm_n_UX_rank = 1 + int(torch.max(significant_modes).item())
            # # # self.pm_n_UX_rank_c_[ncluster] = pm_n_UX_rank

            # this is setting to n_UX_rank (NOT pm_n_UX_rank_c_) SVs regardless of how
            # many are actually used for this cluster; we need this to keep the
            # pm_UX_knc___ tensor padded appropriately.
            # NOTE: This would fail if SVD somehow returned too few for some cluster
            tmp_UX__ = tmp_UX__[0:n_UX_rank, :]
            assert tmp_UX__.shape == (n_UX_rank, grid.n_k_p_r)
            self.pm_UX_knc___[ncluster,:,:] = tmp_UX__


    # this is actually setting up something like a cross-correlation matrix from which one
    # can deduce the information provided by some set of linear combinations of the radii
    # (for each cluster)
    def _determine_principal_modes_empirically(self,
        grid: PolarGrid,
        images: ImageStack
    ) -> Tensor:
        matrix_shape = (self.n_cluster, grid.n_k_p_r, grid.n_k_p_r)
        # TODO: parameterize to allow selecting precision?
        X_2d_Memp_d1_kkc___ = torch.zeros(matrix_shape, dtype=torch.float32)
        X_2d_Memp_d1_weight_rc__ = torch.zeros((self.n_cluster, grid.n_k_p_r), dtype=torch.float32)

        for ncluster in range(self.n_cluster):
            # # # images_this_cluster_ = self.index_nM_from_ncluster__[ncluster]
            # # # cluster_image_count = images_this_cluster_.numel()
            # # # tmp_i8_index_rhs_ = matlab_index_2d_0(
            # # #     grid.n_w_sum,':',
            # # #     images.n_M,images_this_cluster_)
            # # # (
            # # #     X_2d_Memp_d1_kk__,
            # # #     X_2d_Memp_d1_weight_r_,
            # # # ) = principled_marching_empirical_cost_matrix_2(
            # # #     grid.n_k_p_r,
            # # #     grid.k_p_r_,
            # # #     grid.weight_2d_k_p_r_,
            # # #     grid.n_w_,
            # # #     cluster_image_count,
            # # #     torch.reshape(images.M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],(cluster_image_count, grid.n_w_sum)),
            # # # )[:2]
            images_this_cluster__ = images.M_k_p_wkM__[self.index_nM_from_ncluster__[ncluster]]
            cluster_image_count = images_this_cluster__.shape[0]
            (
                X_2d_Memp_d1_kk__,
                X_2d_Memp_d1_weight_r_,
            ) = principled_marching_empirical_cost_matrix_2(
                grid.n_k_p_r,
                grid.k_p_r_,
                grid.weight_2d_k_p_r_,
                grid.n_w_,
                cluster_image_count,
                images_this_cluster__
            )[:2]
            X_2d_Memp_d1_kkc___[ncluster,:,:] = X_2d_Memp_d1_kk__
            X_2d_Memp_d1_weight_rc__[ncluster,:] = X_2d_Memp_d1_weight_r_
        self.pm_X_weight_rc__= X_2d_Memp_d1_weight_rc__
        return X_2d_Memp_d1_kkc___  # AKA pm_X_kkc___


    # this is actually setting up something like a cross-correlation matrix from which one
    # can deduce the information provided by some set of linear combinations of the radii
    # (for each cluster)
    def _determine_principal_modes_from_ansatz(self,
        grid: PolarGrid,
        volume: Volume,
        delta_sigma_base: float = 0.0   # this might also be vector-valued?
    ) -> Tensor:
            matrix_shape = (self.n_cluster, grid.n_k_p_r, grid.n_k_p_r)
            X_2d_xavg_dx_kkc___ = torch.zeros(matrix_shape, dtype=torch.float32)
            X_2d_xavg_dx_weight_rc__ = torch.zeros((self.n_cluster, grid.n_k_p_r), dtype=torch.float32)
            for ncluster in range(self.n_cluster):
                # tmp_CTF_k_p_r_xavg_kk__ = torch.reshape(tmp_CTF_k_p_r_xavg_k_, (1, grid.n_k_p_r)) * torch.reshape(tmp_CTF_k_p_r_xavg_k_, (grid.n_k_p_r, 1))
                isotropic_avg_ctf = self.CTF_k_p_r_xavg_kc__[ncluster]
                tmp_CTF_k_p_r_xavg_kk__ = isotropic_avg_ctf[None, :] * isotropic_avg_ctf[:, None]

                (
                    X_2d_xavg_dx_kk__,
                    X_2d_xavg_dx_weight_r_,
                ) = principled_marching_cost_matrix_7(
                    grid.n_k_p_r,
                    grid.k_p_r_,
                    grid.weight_2d_k_p_r_,
                    volume.l_max_,
                    None,
                    None,
                    volume.a_k_Y_reco_yk_,
                    tmp_CTF_k_p_r_xavg_kk__,
                    delta_sigma_base,
                )[:2]

                X_2d_xavg_dx_kkc___[ncluster,:,:] = X_2d_xavg_dx_kk__
                X_2d_xavg_dx_weight_rc__[ncluster,:] = X_2d_xavg_dx_weight_r_

            self.pm_X_weight_rc__ = X_2d_xavg_dx_weight_rc__
            return X_2d_xavg_dx_kkc___ # aka pm_X_kkc___


    def make_cluster_averages(self, grid: PolarGrid):
        """Create the tensor of "per-cluster representative isotropic CTFs" by
        averaging over the inplane angles of each ctf in each cluster, maintaining
        the radial dimension. Result is stored in self.CTF_k_p_r_xavg_kc__.

        Currently only supports uniform polar grids.

        Args:
            grid (PolarGrid): The polar grid to which the CTFs conform.
        """
        # Sum the per-gridpoint weights for each cluster using a one-hot
        # matrix multiply. The simpler version is:
        #   import torch.nn.functional as F
        #   num_classes = number-of-clusters
        #   oh = F.one_hot(ctfs-to-cluster-map, num_classes=num_classes).T.to(torch.float32)
        #   sum-per-cluster = oh @ ctfs
        # now, that's really sparse, so it'd be ~5x faster to use sparse representation
        # see https://discuss.pytorch.org/t/sum-over-various-subsets-of-a-tensor/31881/8
        # unfortunately, that doesn't work when broadcasting is required, so fall back
        # to the dense version.
        # # # indices = torch.stack((
        # # #         self.index_ncluster_from_nCTF_,
        # # #         torch.arange(self.ctfs.n_CTF, device=self.index_ncluster_from_nCTF_.device)
        # # # ))
        # # # values = torch.ones_like(self.index_ncluster_from_nCTF_, dtype=torch.float32)
        # # # one_hot = torch.sparse_coo_tensor(indices, values, size=(self.n_cluster, self.ctfs.n_CTF))
        # # # # NOTE: RESHAPING ASSUMES UNIFORM GRID (as did original)
        # # # # [IDEA: could maybe do a similar one-hot trick to handle nonuniform grids???]
        # # # weight_sums_per_cluster = torch.mm(one_hot, self.ctfs.CTF_k_p_wkC__)
        num_classes = self.n_cluster
        one_hot = torch.nn.functional.one_hot(self.index_ncluster_from_nCTF_, num_classes=num_classes).T.to(torch.float32)
        weight_sums_per_cluster = (one_hot @ torch.permute(self.ctfs.CTF_k_p_wkC__, (2, 0, 1))).permute(1, 2, 0)
        isotropic = _force_isotropy(weight_sums_per_cluster, grid)
        # Result is clusters x radii (we averaged over the inplanes)
        # divide by per-cluster CTF count, to finish the averaging.
        # (We assume that clusters will always be indexed continuously from 0)
        ctfs_per_cluster_ = torch.bincount(self.index_ncluster_from_nCTF_)
        self.CTF_k_p_r_xavg_kc__ = isotropic / ctfs_per_cluster_[:, None]
    

    def checkpoint_report_clustering(self, params: Parameters):
        params.save_report(
            name_tail="_stage_2.mat",
            data = {
                "pm_UX_knc___": self.pm_UX_knc___,
                # "pm_SX_kc__": self.pm_SX_kc__,
                "pm_n_UX_rank_c_": self.pm_n_UX_rank_c_,},
            min_level = 1
        )
