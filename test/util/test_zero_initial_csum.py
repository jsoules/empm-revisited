import torch
from inspect import getmembers, isfunction, signature, get_annotations
from unittest.mock import Mock, patch
from pytest import raises
from torch.testing import assert_close
from math import sqrt

from empm.util.zero_initial_csum import zero_initial_csum

def test_zero_initial_csum_errors_on_non_int_input():
    input = torch.arange(25) * 1.1
    with raises(ValueError, match='integer vectors'):
        _ = zero_initial_csum(input)


def test_zero_initial_csum_errors_on_higher_dimensional_input():
    input = torch.arange(25).reshape((5, 5))
    with raises(ValueError, match="only defined over vectors"):
        _ = zero_initial_csum(input)


def test_zero_initial_csum():
    input = torch.ones(4, dtype=torch.int32) * 3
    expected = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32)
    res = zero_initial_csum(input)
    assert_close(res, expected)
    assert len(res) == 1 + len(input)
    assert res[0] == 0
