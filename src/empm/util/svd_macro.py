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

# @mark.parametrize("mat", [
#     torch.tensor([[3., 2., 2.,], [2., 3., -2.]]),
#     torch.tensor([[3., 2., 2., 4.,], [2., 4., 3., -2.]]),
# ])
# def test_svd_macro(mat: Tensor):
#     (nat_U, nat_S, nat_V) = torch.linalg.svd(mat, full_matrices=False)
#     (res_U, res_S, res_V) = _svd_macro(mat)

#     assert_close(res_S, nat_S)
#     assert_close(nat_U * -1., res_V.T)
#     assert_close(nat_V * -1., res_U)


# def test_svd_macro_2():
#     loops = 200
#     n_svd = 20

#     for i in range(loops):
#         mat = torch.rand((55, 65)) * (i + 1)
#         print(f"{i}\n{mat}")
#         (nat_U, nat_S, nat_V) = torch.linalg.svd(mat, full_matrices=False)
#         nat_U = nat_U[:, 0:n_svd]
#         nat_S = nat_S[0:n_svd]
#         nat_V = nat_V[0:n_svd, :]
#         (res_U, res_S, res_V) = _svd_macro(mat, n_svd)

#         assert_close(res_S, nat_S)
#         assert_close(nat_U * -1., res_V.T)
#         assert_close(nat_V * -1., res_U)
