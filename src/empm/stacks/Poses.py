from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from torch import tensor, Tensor
from typing_extensions import Self

if TYPE_CHECKING:
    from empm.parameters import Parameters

class Poses():
    """Class tracking currently assigned pose for a collection of images.

    Attributes:
        euler_polar_a_M_ (Tensor): Polar viewing angle, per image
        euler_azimu_b_M_ (Tensor): Azimuthal viewing angle, per image
        euler_gamma_z_M_ (Tensor): Gamma rotational angle, per image
        image_delta_x_acc_M_ (Tensor): Per-image accumulated displacement in x
            (i.e., current image center). This updates only when the upd_M_
            passes a certain threshold. Total accumulated displacement is capped
            by the delta_r_upb parameter; we will force the norm of x_acc_ and
            y_acc not to exceed this value.
        image_delta_y_acc_M_ (Tensor): Per-image accumulated displacement in y
            (i.e., current image center), as per x_acc_.
        image_delta_x_upd_M_ (Tensor): Per-image update to displacement in x
            (i.e., current image shift). This represents the accumulation of
            displacement over several iterations of alignment and reconstruction;
            when it passes a maximum allowed value (set in the delta_r_upd_threshold
            parameter) the _upd_M_ values will be added to _acc_M_ and zeroed out.
        image_delta_y_upd_M_ (Tensor): Per-image update to displacement in y
            (i.e., current image shift). As per x_upd_M_.
        image_delta_x_bit_M_ (Tensor): Per-image increment to displacement update
            in x (calculated per iteration). This represents the displacement update
            from a single iteration of alignment and reconstruction, and will be
            added to the _upd_M_ value then zeroed out after every iteration.
        image_delta_y_bit_M_ (Tensor): Per-image increment to displacement update
            in y (calculated per iteration). As per _y_bit_M_.
        flag_image_delta_upd_M_ (Tensor): Flag identifying which principal-images
            need to be recalculated. That corresponds to those whose _upd_M_
            displacement exceeded the running-total displacement threshold or
            those whose total accumulated displacement _acc_M_ hit the edge
            of the search space defined by delta_r_upb.
        image_I_value_M_ (Tensor): Flag indicating ???
    """

    euler_polar_a_M_: Tensor
    euler_azimu_b_M_: Tensor
    euler_gamma_z_M_: Tensor
    image_delta_x_acc_M_: Tensor
    image_delta_y_acc_M_: Tensor
    image_delta_x_upd_M_: Tensor
    image_delta_y_upd_M_: Tensor
    image_delta_x_bit_M_: Tensor
    image_delta_y_bit_M_: Tensor
    flag_image_delta_upd_M_: Tensor
    image_I_value_M_: Tensor

    def __init__(self,
        n_imgs: int = -1,
        *,
        euler_polar_a_M_: Tensor | None = None,
        euler_azimu_b_M_: Tensor | None = None,
        euler_gamma_z_M_: Tensor | None = None,
        image_delta_x_acc_M_: Tensor | None = None,
        image_delta_y_acc_M_: Tensor | None = None,
        image_delta_x_upd_M_: Tensor | None = None,
        image_delta_y_upd_M_: Tensor | None = None,
        flag_image_delta_upd_M_: Tensor | None = None,
        image_I_value_M_: Tensor | None = None
    ):
        args = dict(locals())
        empirical_n_imgs = None if n_imgs < 0 else n_imgs
        for x in args.keys():
            this_param = args[x]
            if isinstance(this_param, Tensor):
                this_array_length = len(this_param)
                if empirical_n_imgs is None:
                    empirical_n_imgs = this_array_length
                if this_array_length != empirical_n_imgs:
                    raise ValueError("Attempt to initialize Poses with inconsistently-numbered non-empty inputs.")
        if empirical_n_imgs is None:
            raise ValueError("Attempt to create Poses object without specifying total image count.")
        
        # This is really ugly, there's gotta be a better way than listing each of these manually
        euler_polar_a_M_ = euler_polar_a_M_ if euler_polar_a_M_ is not None else torch.zeros(empirical_n_imgs)
        euler_azimu_b_M_ = euler_azimu_b_M_ if euler_azimu_b_M_ is not None else torch.zeros(empirical_n_imgs)
        euler_gamma_z_M_ = euler_gamma_z_M_ if euler_gamma_z_M_ is not None else torch.zeros(empirical_n_imgs)
        image_delta_x_acc_M_ = image_delta_x_acc_M_ if image_delta_x_acc_M_ is not None else torch.zeros(empirical_n_imgs)
        image_delta_y_acc_M_ = image_delta_y_acc_M_ if image_delta_y_acc_M_ is not None else torch.zeros(empirical_n_imgs)
        image_delta_x_upd_M_ = image_delta_x_upd_M_ if image_delta_x_upd_M_ is not None else torch.zeros(empirical_n_imgs)
        image_delta_y_upd_M_ = image_delta_y_upd_M_ if image_delta_y_upd_M_ is not None else torch.zeros(empirical_n_imgs)
        image_I_value_M_ = image_I_value_M_ if image_I_value_M_ is not None else torch.ones(empirical_n_imgs)
        flag_image_delta_upd_M_ = flag_image_delta_upd_M_ if flag_image_delta_upd_M_ is not None else torch.ones(empirical_n_imgs)
        
        self.euler_polar_a_M_ = euler_polar_a_M_.to(torch.float32)
        self.euler_azimu_b_M_ = euler_azimu_b_M_.to(torch.float32)
        self.euler_gamma_z_M_ = euler_gamma_z_M_.to(torch.float32)
        self.image_delta_x_acc_M_ = image_delta_x_acc_M_.to(torch.float32)
        self.image_delta_y_acc_M_ = image_delta_y_acc_M_.to(torch.float32)
        self.image_delta_x_upd_M_ = image_delta_x_upd_M_.to(torch.float32)
        self.image_delta_y_upd_M_ = image_delta_y_upd_M_.to(torch.float32)
        self.image_I_value_M_ = image_I_value_M_.to(torch.float32)
        self.flag_image_delta_upd_M_ = flag_image_delta_upd_M_.to(torch.int32)

        self.image_delta_x_bit_M_ = torch.zeros(empirical_n_imgs, dtype=torch.float32)
        self.image_delta_y_bit_M_ = torch.zeros(empirical_n_imgs, dtype=torch.float32)


    @classmethod
    def random_init(cls, n_imgs: int) -> Self:
        euler_polar_a_M_ = 1 * torch.pi * torch.rand(n_imgs, dtype=torch.float32)
        euler_azimu_b_M_ = 2 * torch.pi * torch.rand(n_imgs, dtype=torch.float32)
        euler_gamma_z_M_ = 2 * torch.pi * torch.rand(n_imgs, dtype=torch.float32)

        return cls(
            n_imgs,
            euler_polar_a_M_=euler_polar_a_M_,
            euler_azimu_b_M_=euler_azimu_b_M_,
            euler_gamma_z_M_ = euler_gamma_z_M_
        )
    

    def update_translations(self, parameter: Parameters, machine_tolerance: float = 1e-6):
        # in current implementation, as threshold is 0, we just update upd -> acc every loop
        # (that obviously also includes adding bit to upd first)

        # NOTE: future versions may want to interpolate somewhat here
        self.image_delta_x_upd_M_ += self.image_delta_x_bit_M_
        self.image_delta_y_upd_M_ += self.image_delta_y_bit_M_

        # equivalent to, but shorter than, torch.linalg.norm(torch.stack((image_delta_x_upd_M_, image_delta_y_upd_M_)), dim=0, keepdim=True)
        image_delta_r_upd_norm_M_ = torch.sqrt(self.image_delta_x_upd_M_**2 + self.image_delta_y_upd_M_**2)

        # Apply hard cap to upd values if (acc + upd) > hard limit of upb
        image_delta_x_tot_M_ = self.image_delta_x_acc_M_ + self.image_delta_x_upd_M_
        image_delta_y_tot_M_ = self.image_delta_y_acc_M_ + self.image_delta_y_upd_M_
        image_delta_r_tot_M_ = torch.sqrt(image_delta_x_tot_M_**2 + image_delta_y_tot_M_**2)

        # now, for those cases where the total radius exceeds the maximum allowable, shrink the max by the ratio
        # 1-hot mask of indices whose displacement exceeds the allowable total
        hard_upper_bound_mask = torch.zeros_like(
            image_delta_r_upd_norm_M_, dtype=torch.int32
        ).scatter_(0, torch.where(image_delta_r_tot_M_ > parameter.delta_r_upb)[0], 1)
        not_upper_bounded = torch.logical_not(hard_upper_bound_mask)
        # and a mask for those images whose upd displacement exceeds the limit but without hitting the
        # hard-upper-bound cap.
        soft_upper_bound_mask = torch.zeros_like(
            image_delta_r_upd_norm_M_, dtype=torch.int32
        ).scatter_(0, torch.where(image_delta_r_upd_norm_M_ >= parameter.delta_r_upd_threshold)[0], 1)
        soft_upper_bound_mask *= not_upper_bounded

        # if there are any indices whose final value, acc + upd, exceeds the max allowable,
        # handle them here
        if torch.any(hard_upper_bound_mask > 0):
            compression_factor = parameter.delta_r_upb / torch.maximum(image_delta_r_tot_M_, torch.tensor(machine_tolerance))
            # vector of 1s for non-compressed indices and (ratios) for indices to be compressed
            # (adding back the not_upper_bounded sets 0ed indices to 1)
            final_compression = compression_factor * hard_upper_bound_mask + not_upper_bounded
            self.image_delta_x_acc_M_ *= final_compression
            self.image_delta_y_acc_M_ *= final_compression
        
        self.image_delta_x_acc_M_ += self.image_delta_x_upd_M_ * soft_upper_bound_mask
        self.image_delta_y_acc_M_ += self.image_delta_y_upd_M_ * soft_upper_bound_mask

        all_changed_mask = hard_upper_bound_mask + soft_upper_bound_mask
        self.flag_image_delta_upd_M_ = all_changed_mask.to(torch.int32)
        # zero out all the bits, and the upds which participated in an acc update
        self.image_delta_x_upd_M_ *= torch.logical_not(all_changed_mask)
        self.image_delta_y_upd_M_ *= torch.logical_not(all_changed_mask)
        self.image_delta_x_bit_M_ = torch.zeros_like(self.image_delta_x_upd_M_)
        self.image_delta_y_bit_M_ = torch.zeros_like(self.image_delta_y_upd_M_)

