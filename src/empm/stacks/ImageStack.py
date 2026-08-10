from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

from dir_empm.interp_x_c_to_k_p_xxnufft import interp_x_c_to_k_p_xxnufft

if TYPE_CHECKING:
    from .Poses import Poses
    from empm.grids import PolarGrid

class ImageStack():
    n_M: int
    M_k_p_wkM__: Tensor

    def __init__(self,
        M_k_p_wkM__: Tensor
    ):
        # TODO: any sort of consistency check
        # TODO: Something about loading functionality?
        self.M_k_p_wkM__ = M_k_p_wkM__.to(torch.complex64)
        self.n_M = M_k_p_wkM__.shape[0]


    def apply_displacements_from_poses(self, grid: PolarGrid, poses: Poses, scratch: Tensor | None = None) -> Tensor:
        if scratch is None:
            # allocate uninitialized rather than doing ones-like
            # scratch = torch.ones_like(self.M_k_p_wkM__, dtype=torch.complex64)
            scratch = torch.tensor(self.M_k_p_wkM__.size(), dtype=torch.complex64)
        
        if grid.is_uniform:
            L_c_wkv__ = \
                grid.k_c_0_wk_ * poses.image_delta_x_acc_M_[:, None] \
                + grid.k_c_1_wk_ * poses.image_delta_y_acc_M_[:, None]
            ## ADDITION: Restricts to affected images by zeroing out non-flagged elements
            L_c_wkv__ *= poses.flag_image_delta_upd_M_[:, None]
            C_c_wkv__ = torch.exp(-1j * 2 * torch.pi * L_c_wkv__).to(dtype=torch.complex64)
            # NOTE: CONFIRM: that self.M_k_p_wkM__ is already image_idx x flat_points
            torch.mul(C_c_wkv__, self.M_k_p_wkM__, out=scratch)
            return scratch

        else:
            raise NotImplementedError()
            # This entire branch shouldn't be needed; we have the vectorized k_c_s for the whole grid
            # (even for adaptive grids) so the above branch should Just Work


    # TODO NOTE: the cartesian grid could also be an object, though we aren't using it anywhere else yet
    @classmethod
    def from_cartesian_image_stack(cls,
        image_tensor: Tensor,
        n_x1: int,
        n_x2: int,
        diameter_x1_c: float,
        diameter_x2_c: float,
        polar_grid: PolarGrid,
        source_is_centered: bool = True
    ):
        """Create an image stack suitable for this package (Fourier-space representations
        on polar grid) from a physical-space representation on a Cartesian grid.

        Note this DOES NOT NORMALIZE the result. since such normalization is actually
        irrelevant for downstream EMPM processing.

        Args:
            image_tensor (Tensor): Tensor of image intensities, addressed as
                [image_index X dimension-1 X dimension-2]
            n_x1 (int): Number of points in the dimension-1 dimension of the Cartesian grid
            n_x2 (int): Number of points in the dimension-2 dimension of the Cartesian grid
            diameter_x1_c (float): Total span in dimension 1 of the Cartesian grid (Angstrom)
            diameter_x2_c (float): Total span in dimension 2 of the Cartesian grid (Angstrom)
            polar_grid (PolarGrid): Target polar grid to fit
            source_is_centered (bool): Whether the source images are on a centered grid (the
                default) or an uncentered one
        """
        n_M = image_tensor.shape[0]
        M_k_p_wkM__ = torch.zeros((n_M, polar_grid.n_w_sum), dtype=torch.complex64)
        for nM in range(n_M):
            img = interp_x_c_to_k_p_xxnufft(
                n_x1,
                diameter_x1_c,
                n_x2,
                diameter_x2_c,
                image_tensor[nM, :, :],
                polar_grid.n_k_p_r,
                polar_grid.k_p_r_,
                polar_grid.n_w_,
                0 if source_is_centered else 1  # aka flag_u_vs_c
            )
            M_k_p_wkM__[nM, :] = img

        return cls(M_k_p_wkM__)

# TODO: NOTE: This is a vectorized way of setting up k_c_s for uniform grids.
# (For reference elsewhere)

# get each inplane rotation, but don't repeat the 2pi at the end
# sadly, pytorch linspace doesn't offer an option for exclusive spacing
# gamma_z_ = torch.linspace(0, 2 * torch.pi, grid.n_w_max + 1, dtype=torch.float32)[:-1]

# k_c_0_wk_ = (torch.cos(gamma_z_)[None, :] * grid.k_p_r_[:, None]).ravel()
# k_c_1_wk_ = (torch.sin(gamma_z_)[None, :] * grid.k_p_r_[:, None]).ravel()

