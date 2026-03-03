from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

from empm.parameters import MACHINE_TOLERANCE
from empm.util import matlab_style_svd_macro

if TYPE_CHECKING:
    from empm.parameters import Parameters
    from empm.grids import PolarGrid


class CTF():
    """Class representing a stack of CTFs (contrast transfer functions),
    in Fourier space.
    """

    n_CTF: int
    CTF_k_p_wkC__: Tensor
    index_nCTF_from_nM_: Tensor

    # CTF_k_p_wkC__ is the actual CTF collection, each one
    # represented in Fourier space as linearized points,
    # inplane angle (w) changing faster, radial dimension (k)
    # changing slower.
    # Presumably this invokes some kind of grid; checking that
    # might be a good idea, but idk if we really want to attach
    # it to this object?

    # TODO: Actually can we build this from the image source files or something?
    def __init__(self, CTF_k_p_wkC__: Tensor, index_nCTF_from_nM_: Tensor):
        raise NotImplementedError()
        ...


    # TODO: have I got this right?
    def expand_single_value_isotropic_ctf(self, grid: PolarGrid):
        if not grid.is_uniform:
            # NOTE: Does this work for non-uniform grids?
            # My instrinct is that it would fail for non-uniform...
            pass
        if (self.CTF_k_p_wkC__.shape[-1] == grid.n_k_p_r):
            # by construction, we are working with a tensor of shape
            # [n_ctf x n_k_p_r]
            # This is an isotropic CTF representing each the weight for each
            # point on each ring as a single value per radial coordinate.
            # We'd like to expand this so that we materialize the values of the
            # tensor at each grid point, which means adding a lowest-order (inplane
            # rotation) dimension and duplicating across it.
            self.CTF_k_p_wkC__ = \
                torch.reshape(
                    torch.tile(self.CTF_k_p_wkC__[:,:,None], (1, 1, grid.n_w_max)),
                    (-1, grid.n_w_sum)
                )


    # NOTE: This is very similar to operations in CTFCluster to
    # determine per-cluster principal modes (determine_principal_modes)
    def empirically_determine_rank(self, parameter: Parameters, grid: PolarGrid, n_M: int) -> int:
        # NOTE: This is probably not necessary.
        # Indexing step would find the indices corresponding to all points,
        # and any CTF index that appears in the images-to-ctfs index.
        # But we expect the latter map to be bijective, as each image should have its
        # own unique CTF, esp. once we've broadcast isotropic CTFs.
        # So this only does something in the case where we have CTFs in the tensor
        # that aren't actually used by any image, in which case we're better off just
        # dropping those from the tensor or something...

        # # # tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',
        # # #                                       n_CTF,index_nCTF_from_nM_)
        
        # Now reshaping CTF_k_p_wkC__ to (n_images x n_total points) is unnecessary as
        # the CTF tensor already has that shape, EXCEPT again in the case where n_CTF
        # does not match n_M, in which case this would fail anyway.
        # Just to be sure we'll assert it.

        # # # _, SCTF_ ,_ = matlab_svds(torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_],
        # # #                                       (n_M, n_w_sum)),
        # # #                                       int(np.minimum(n_w_sum,n_M)))
        
        assert self.CTF_k_p_wkC__.shape == (n_M, grid.n_w_sum)
        max_rank = min(grid.n_w_sum, n_M)
        _, SCTF_ ,_ = matlab_style_svd_macro(self.CTF_k_p_wkC__, max_rank)
        # TODO QUERY: Honestly though, if the biggest one isn't over machine tolerance,
        # isn't that its own sort of problem?
        divisor = max(MACHINE_TOLERANCE, SCTF_[0])  # as S is in desc order, the first one is the max
        rank = int((SCTF_ / divisor > parameter.tolerance_master).sum().item())
        # as written, would find the indices of all S-values from SVD which are non-negligible,
        # then add 1 to the maximum index, which (as they're sorted and 0-indexed) yields a count.
        # But easier just to count them
        return rank
