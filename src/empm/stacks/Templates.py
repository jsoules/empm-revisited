import torch
from torch import Tensor

# TODO: Might be better to have the constructor allocate based on the grid, and then let
# others reuse the allocated memory. But the logic for this is pretty deep, I think in
# sample_shell_6.py, and pulling it out would be nontrivial.

class Templates():
    S_k_p_wkS__: Tensor
    n_S: int
    viewing_azimu_b_S_: Tensor
    viewing_polar_a_S_: Tensor


    def __init__(self, n_S: int, S_k_p_wkS__: Tensor, viewing_azimu_b_s: Tensor, viewing_polar_a_S_: Tensor):
        self.S_k_p_wkS__ = S_k_p_wkS__
        self.n_S = n_S
        self.viewing_azimu_b_S_ = viewing_azimu_b_s
        self.viewing_polar_a_S_ = viewing_polar_a_S_
