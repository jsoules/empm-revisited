import torch
from torch import Tensor

# In this example, limiting by n_svd doesn't seem to do anything b/c
# those are already the nonzero singular values, but
# maybe it's different in the realistic case?
def matlab_style_svd_macro(m: Tensor, n_svd: int = -1) -> tuple[Tensor, Tensor, Tensor]:
    if n_svd < 0:
        n_svd = min(m.shape)
    _U__, _S_, _V__ = torch.linalg.svd(m.T, full_matrices=False)
    _U__ = _U__.T
    # Not sure if this does anything in realistic cases?
    _U__ = _U__[0:n_svd, :]
    _S_ = _S_[0:n_svd]
    _V__ = _V__[0:n_svd, :]
    return (_U__, _S_, _V__)
