from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Poses import Poses
    from empm.grids import PolarGrid

class ImageStack():
    n_M: int
    M_k_p_wkM__: Tensor

    def __init__(self):
        raise NotImplementedError()
    
    def apply_displacements_from_poses(self, grid: PolarGrid, poses: Poses, scratch: Tensor | None = None) -> Tensor:
        if scratch is None:
            scratch = torch.ones_like(self.M_k_p_wkM__, dtype=torch.complex64)
        
        # TODO: Confirm delta-x and delta-y are same size (that check would belong in the Poses class)
        if grid.is_uniform:
            L_c_wkv__ = \
                grid.k_c_0_wk_ * poses.image_delta_x_acc_M_[:, None] \
                + grid.k_c_1_wk_ * poses.image_delta_y_acc_M_[:, None]
            ## ADDITION: Restricts to affected images by zeroing out non-flagged elements
            L_c_wkv__ = L_c_wkv__ * poses.flag_image_delta_upd_M_[:, None]
            C_c_wkv__ = torch.exp(-1j * 2 * torch.pi * L_c_wkv__).to(dtype=torch.complex64)
            # NOTE: CONFIRM: that self.M_k_p_wkM__ is already image_idx x flat_points
            torch.mul(C_c_wkv__, self.M_k_p_wkM__, out=scratch)
            # TODO: check with assert whether this is necessary
            scratch = scratch.to(torch.complex64)
            return scratch

        else:
            raise NotImplementedError()
            # This entire branch shouldn't be needed; we have the vectorized k_c_s for the whole grid
            # (even for adaptive grids) so the above branch should Just Work

# TODO: NOTE: This is a vectorized way of setting up k_c_s for uniform grids.
# (For reference elsewhere)

# get each inplane rotation, but don't repeat the 2pi at the end
# sadly, pytorch linspace doesn't offer an option for exclusive spacing
# gamma_z_ = torch.linspace(0, 2 * torch.pi, grid.n_w_max + 1, dtype=torch.float32)[:-1]

# k_c_0_wk_ = (torch.cos(gamma_z_)[None, :] * grid.k_p_r_[:, None]).ravel()
# k_c_1_wk_ = (torch.sin(gamma_z_)[None, :] * grid.k_p_r_[:, None]).ravel()

