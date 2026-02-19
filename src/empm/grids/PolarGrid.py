import numpy as np
import torch
from torch import Tensor
# from typing import Self  # note this requires python >= 3.11;
# for python 3.10 use
from typing_extensions import Self

## TODO: Import get_weight_2d_2 properly
# stub here to keep my linter happy
def get_weight_2d_2(
        flag_verbose: int = 0,
        n_k_p_r: int = 0,
        k_p_r_: Tensor = torch.zeros(0),
        k_p_r_max: int = 0,
        template_k_eq_d: float = -1.,
        n_w_0in_: Tensor = torch.zeros(0),
        weight_3d_k_p_r_: Tensor = torch.zeros(0)
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # # #         n_w_,
    # # #         weight_2d_k_p_r_,
    # # #         weight_2d_k_p_wk_,
    # # #         k_p_r_wk_,
    # # #         k_p_w_wk_,
    # # #         k_c_0_wk_,
    # # #         k_c_1_wk_,
    ...

class PolarGrid():
    """Class implementing 2D polar-coordinate grid.

    Attributes:
        is_uniform (bool): Flag indicating whether the grid is uniform
        n_k_p_r (int): Number of frequency bands / radii
        k_p_r_ (Tensor): Frequency (k) value for each band. 1d real tensor, expected
            to have n_k_p_r elements.
        k_p_r_max (int): Highest frequency (k) value on the polar grid. Should be
            equal to the maximum value of k_p_r_.
        template_k_eq_d (float): Equatorial distance between frequencies for a
            uniform grid [??]
        n_w_ (Tensor): Number of angles per shell (will be the same value for
            every r in the case of a uniform grid). 1d integral tensor of length
            equal to n_k_p_r.
        weight_2d_k_p_r_ (Tensor): Quadrature-weight for each frequency band
            on a 2d polar grid. 1D real tensor of length equal to the number
            of frequency bands, n_k_p_r. Sums to pi * k_p_r_max^2.
        weight_2d_k_p_wk_ (Tensor): Quadrature-weight for each quadrature
            point on the grid. Unrolled 1D real tensor of total length equal to
            the number of quadrature points, which is n_w_[0] * n_k_p_r for
            uniform grids. For uniform grids, the nth element of this tensor
            represents the weight at the point whose radial dimension is
            floor(n / n_w_[0]) and whose rotational dimension is n % n_w_[0].
            Sums to k_p_r_max^2 / 4pi.
        k_p_r_wk_ (Tensor): Radial dimension of each quadrature point on the
            grid, unrolled to 1D. 1D real tensor of total length equal
            to the number of quadrature points (n_w_[0] * n_k_p_r for uniform
            grids). In the case of uniform grids, the first n_w_[0] elements
            will equal k_p_r_[0], the second n_w_[0] elements will equal
            k_p_r_[1], etc.
        k_p_w_wk_ (Tensor): Angular dimension of each quadrature point on the
            grid, unrolled to 1D. 1D real tensor of total length equal to the
            number of quadrature points (n_w_[0] * n_k_p_r for uniform grids).
            In the case of uniform grids, subsequent elements of this tensor
            will increase by steps of size template_k_eq_d until they reach
            2pi, at which point they'll start over from 0; there should be
            n_k_p_r such repetitions.
        k_c_0_wk_ (Tensor): Cartesian x-coordinate of each quarature point on
            the grid, unrolled to 1D. 1D real tensor of total length equal to
            the number of quadrature points (as above). For an element n
            of this tensor, k_c_0_wk_[n] = k_p_r_wk_[n] * cos(k_p_w_wk_[n]).
        k_c_1_wk_ (Tensor): Cartesian y-coordinate of each quarature point on
            the grid, unrolled to 1D. 1D real tensor of total length equal to
            the number of quadrature points (as above). For an element n
            of this tensor, k_c_0_wk_[n] = k_p_r_wk_[n] * sin(k_p_w_wk_[n]).
        n_w_sum (int): Total number of points across all radii
        n_w_max (int): Highest number of radial points for any radius. For
            regular grids, this should be equal to n_w_sum times n_k_p_r.
        n_w_csum_ (Tensor): Index of the first point in each frequency ring
             in the grid. Assume the grid points are numbered linearly starting
             from the innermost frequency ring and incrementing the inplane angle
             before incrementing the radial coordinate. In such a linearized
             numbering scheme, the index of the first grid point for the nth
             radius will be at n_w_csum_[n].
    """

    is_uniform: bool
    n_k_p_r: int
    k_p_r_: Tensor
    k_p_r_max: int
    template_k_eq_d: float
    n_w_: Tensor
    weight_2d_k_p_r_: Tensor
    weight_2d_k_p_wk_: Tensor
    k_p_r_wk_: Tensor
    k_p_w_wk_: Tensor
    k_c_0_wk_: Tensor
    k_c_1_wk_: Tensor
    n_w_sum: int
    n_w_max: int
    n_w_csum_: Tensor


    def __init__(self,
        is_uniform: bool,
        n_k_p_r: int,
        k_p_r_: Tensor,
        k_p_r_max: int,
        template_k_eq_d: float,
        n_w_: Tensor,
        weight_2d_k_p_r_: Tensor,
        weight_2d_k_p_wk_: Tensor,
        k_p_r_wk_: Tensor,
        k_p_w_wk_: Tensor,
        k_c_0_wk_: Tensor,
        k_c_1_wk_: Tensor,
    ) -> None:
        self.is_uniform = is_uniform
        self.n_k_p_r = n_k_p_r
        self.k_p_r_ = k_p_r_
        self.k_p_r_max = k_p_r_max
        self.template_k_eq_d = template_k_eq_d
        self.n_w_ = n_w_.ravel()    # ensure 1D
        self.weight_2d_k_p_r_ = weight_2d_k_p_r_
        self.weight_2d_k_p_wk_ = weight_2d_k_p_wk_
        self.k_p_r_wk_ = k_p_r_wk_
        self.k_p_w_wk_ = k_p_w_wk_
        self.k_c_0_wk_ = k_c_0_wk_
        self.k_c_1_wk_ = k_c_1_wk_

        self.n_w_sum = int(torch.sum(self.n_w_).item())
        self.n_w_max = int(torch.max(n_w_).item())
        self.n_w_csum_ = torch.cumsum(
            torch.concatenate((torch.tensor([0]), self.n_w_)), 0
        ).to(torch.int32)
        if is_uniform:
            assert len(torch.unique(self.n_w_)) == 1
            assert self.n_w_max % 2 == 0, "Odd number of inplane rotations for grid"
            assert self.n_w_sum == self.n_w_max * self.n_k_p_r


    @classmethod
    def make_uniform_grid(
        cls,
        n_k_p_r: int = 0,
        k_p_r_: Tensor = torch.zeros(0),
        k_p_r_max: int = 0,
        template_k_eq_d: float = -1.,
        n_w_0in_: Tensor = torch.zeros(0),
        weight_3d_k_p_r_: Tensor = torch.zeros(0)
    ) -> Self:
        # NOTE there are shortcuts for this sort of long list, there's
        # likely a better way to do this
        # But I don't want to introduce any more complication than I need to

        (n_w_,
        weight_2d_k_p_r_,
        weight_2d_k_p_wk_,
        k_p_r_wk_,
        k_p_w_wk_,
        k_c_0_wk_,
        k_c_1_wk_,
        ) = get_weight_2d_2(
            flag_verbose = 0,
            n_k_p_r = n_k_p_r,
            k_p_r_ = k_p_r_,
            k_p_r_max = k_p_r_max,
            template_k_eq_d = template_k_eq_d,
            n_w_0in_ = n_w_0in_,
            weight_3d_k_p_r_ = weight_3d_k_p_r_,
        )

        return cls(
            is_uniform = True,
            n_k_p_r = n_k_p_r,
            k_p_r_ = k_p_r_,
            k_p_r_max = k_p_r_max,
            template_k_eq_d = template_k_eq_d,
            n_w_ = n_w_,
            weight_2d_k_p_r_ = weight_2d_k_p_r_,
            weight_2d_k_p_wk_ = weight_2d_k_p_wk_,
            k_p_r_wk_ = k_p_r_wk_,
            k_p_w_wk_ = k_p_w_wk_,
            k_c_0_wk_ = k_c_0_wk_,
            k_c_1_wk_ = k_c_1_wk_,
        )
