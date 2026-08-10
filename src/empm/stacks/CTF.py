from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING
from math import sqrt

from empm.parameters import MACHINE_TOLERANCE
from empm.util import matlab_style_svd_macro

if TYPE_CHECKING:
    from empm.parameters import Parameters
    from empm.grids import PolarGrid

from dir_empm.niko_ctf import niko_ctf


class CTF():
    """Class representing a stack of CTFs (contrast transfer functions),
    in Fourier space.

    Attributes:
        n_CTF (int): Total number of CTFs in the stack. Should equal
            the first dimension of the CTF tensor.
        CTF_k_p_wkC__ (Tensor): The actual CTF collection, each CTF
            represented in Fourier space as linearized points, with the
            inplane angle (w) changing faster, radial dimension (k) changing
            slower. However, isotropic CTF tensors may be represented by a
            single value per radial dimension, in which case this tensor
            would be [n_CTF x n_radii] instead of [n_CTF x n_grid_points].
            To correct this, run the expand_single_value_isotropic_ctf
            function before using elsewhere.
        index_nCTF_from_nM_ (Tensor): Vector mapping indices in the image
            tensor (M) to the CTF belonging to that image (CTF)
    """
    n_CTF: int
    CTF_k_p_wkC__: Tensor
    index_nCTF_from_nM_: Tensor

    # TODO: Actually can we build this from the image source files or something?
    def __init__(self, CTF_k_p_wkC__: Tensor, index_nCTF_from_nM_: Tensor):
        # TODO: More consistency checks
        self.n_CTF = CTF_k_p_wkC__.shape[0]
        self.CTF_k_p_wkC__ = CTF_k_p_wkC__
        self.index_nCTF_from_nM_ = index_nCTF_from_nM_


    # TODO: have I got this right?
    def expand_single_value_isotropic_ctf(self, grid: PolarGrid):
        """Consumers of the CTFs will assume there is one value per grid point.
        In the case where the (isotropic) CTF tensors are represented as a single
        weight value per radial distance, this code will expand those single values
        to bring the CTF shape in line with the shape of the image tensors. Currently
        only defined over uniform grids.

        Args:
            grid (PolarGrid): The desired target grid
        """
        if not grid.is_uniform:
            # NOTE: Does this work for non-uniform grids?
            # My instrinct is that it would fail for non-uniform...
            pass
        if (self.CTF_k_p_wkC__.shape[-1] == grid.n_k_p_r):
            # [n_ctf x n_k_p_r] --> [n_ctf x (n_k_p_r * n_w)]
            self.CTF_k_p_wkC__ = \
                torch.reshape(
                    torch.tile(self.CTF_k_p_wkC__[:,:,None], (1, 1, grid.n_w_max)),
                    (-1, grid.n_w_sum)
                )


    # Note similarity to CTFCluster's determining-principal-modes code.
    def empirically_determine_rank(self, parameter: Parameters, grid: PolarGrid, n_M: int) -> int:
        """Computes an estimate of the rank of the CTFs, which is used in determining cluster
        counts.

        Args:
            parameter (Parameters): Shared Parameters object defining acceptable tolerances
            grid (PolarGrid): Grid object defining number of grid points per CTF/Image
            n_M (int): Number of images in the stack (to ensure CTF stack is of the same
                dimension as the Image stack)

        Returns:
            int: Estimated rank of the collection of CTFs
        """
        # Ensure the CTF count matches the image count, i.e. we don't have any unused CTFs
        # floating around in the CTF tensor.
        assert self.CTF_k_p_wkC__.shape == (n_M, grid.n_w_sum)
        max_rank = min(grid.n_w_sum, n_M)
        _, SCTF_ ,_ = matlab_style_svd_macro(self.CTF_k_p_wkC__, max_rank)
        # TODO QUERY: Honestly though, if the biggest one isn't over machine tolerance,
        # isn't that its own sort of problem?
        divisor = max(MACHINE_TOLERANCE, SCTF_[0])  # as S is in desc order, the first one is the max
        rank = int((SCTF_ / divisor > parameter.tolerance_master).sum().item())
        return rank


    @classmethod
    def make_ctfs_from_parameters(cls,
        n_CTF: int,
        grid: PolarGrid,
        n_pixels_across: int,
        pixel_size_angstrom: float,
        voltage_C_: Tensor | list[float],
        defocusU_C_: Tensor | list[float],
        defocusV_C_: Tensor | list[float],
        defocusAngle_C_: Tensor | list[float],
        sphericalAberration_C_: Tensor | list[float],
        amplitudeContrast_C_: Tensor | list[float],
        defocus_is_radians: bool = False,
        index_nCTF_from_nM_: Tensor | None = None
    ):
        """Generate a CTF stack for a particular polar grid from CTF parameters.

        Args:
            n_CTF (int): Number of CTFs to generate
            grid (PolarGrid): Fourier-space polar grid object to fit
            n_pixels_across (int): Dimension of the (presumed square) box
            pixel_size_angstrom (float): size of detector pixels, in physical space, in Angstrom
            voltage_C_ (Tensor | list[float]): voltage value per CTF, in kV
            defocusU_C_ (Tensor | list[float]): defocus-u value per CTF, in Angstrom
            defocusV_C_ (Tensor | list[float]): defocus-v value per CTF, in Angstrom
            defocusAngle_C_ (Tensor | list[float]): defocus angle/astigmatism angle per CTF
            sphericalAberration_C_ (Tensor | list[float]): lens spherical aberration per CTF, in mm
            amplitudeContrast_C_ (Tensor | list[float]): amplitude contrast per CTF
            defocus_is_radians (bool): Whether the defocus angle is in radians
            index_nCTF_from_nM_ (Tensor | None, optional): Mapping of CTFs to image stack. Defaults to None.
                If not provided, we will assume images and CTFs have the same index.
        """
        CTF_k_p_wkC__ = torch.zeros((n_CTF, grid.n_w_sum), dtype=torch.float64)
        for n in range(n_CTF):
            # convert voltage from kV to V, then voltage into electron wavelength in A
            _voltage_1V = voltage_C_[n] * 1000.0
            _lambda = 12.2643247 / max(1e-24, sqrt(_voltage_1V + _voltage_1V**2 * 0.978466e-6))
            _angle = defocusAngle_C_[n] if defocus_is_radians else (defocusAngle_C_[n] * torch.pi)/180.0
            _amplitude_contrast = amplitudeContrast_C_[n]

            # object pixel size = Detector_Pixel_Size / max(1e-24, CTF_Magnification) <-- TODO: is this an out-of-date note?
            box_size_ang = n_pixels_across * pixel_size_angstrom
            # TODO: Q: Wouldn't a box size below a tiny fraction of an angstrom indicate an error state?
            _lambda_per_box = _lambda / max(1e-24, box_size_ang)
            CTF_k_p_wkC__[n,:] = -niko_ctf(
                    Cs = sphericalAberration_C_[n] * 1.0e7,
                    var_lambda = _lambda,
                    w1 = sqrt(1.0 - _amplitude_contrast ** 2),
                    w2 = _amplitude_contrast,
                    df1 = defocusU_C_[n],
                    df2 = defocusV_C_[n],
                    angast = _angle,
                    thetatr = _lambda_per_box/torch.pi,
                    l_ = grid.k_c_0_wk_,
                    m_ = grid.k_c_1_wk_
                )[0]
        image_mapping = index_nCTF_from_nM_ if index_nCTF_from_nM_ is not None else torch.arange(n_CTF)
        return cls(CTF_k_p_wkC__, image_mapping)
