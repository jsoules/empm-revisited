import torch
from torch import Tensor

def zero_initial_csum(input: Tensor) -> Tensor:
    if len(input.shape) != 1:
        raise ValueError("This operation is only defined over vectors.")
    if input.dtype != torch.int32 and input.dtype != torch.int64:
        raise ValueError("This operation is only defined over integer vectors.")

    csum = torch.cumsum(input, 0)
    result = torch.concatenate((torch.tensor([0]), csum)).to(torch.int32)
    return result
