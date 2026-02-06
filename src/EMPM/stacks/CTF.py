import torch
from torch import Tensor
from ..grids import PolarGrid

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
