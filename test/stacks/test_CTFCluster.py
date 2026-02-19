import torch
from torch.testing import assert_close
from unittest.mock import Mock
from pytest import raises

from empm.stacks import force_isotropy


def test_force_isotropy_fails_for_nonuniform():
    n_angles = 7
    n_rings = 4
    mock_ctf = Mock()

    grid = Mock()
    grid.n_k_p_r = n_rings
    grid.n_w_max = n_angles
    grid.is_uniform = False

    with raises(Exception, match="varying inplane angle"):
        _ = force_isotropy(mock_ctf, grid)


def test_force_isotropy():
    n_angles = 7
    n_rings = 4
    n_ctfs = 5
    per_ring_vals = torch.arange(n_angles) + 1.
    ring_multipliers = torch.arange(n_rings) + 1.
    ctf_multipliers = torch.arange(n_ctfs) + 1.
    base_ctf = per_ring_vals[None, :] * ring_multipliers[:, None]
    ctf_vals = base_ctf[None, :, :] * ctf_multipliers[:, None, None]
    # result is n_ctfs x n_rings x n_angles, where each ring is an
    # integer multiple of the previous & each CTF is an integer multiple
    # of the previous
    mock_ctf = Mock()
    mock_ctf.CTF_k_p_wkC__ = ctf_vals

    grid = Mock()
    grid.n_k_p_r = n_rings
    grid.n_w_max = n_angles
    grid.is_uniform = True

    res = force_isotropy(mock_ctf, grid)

    assert res.shape == (n_ctfs, n_rings)
    expected = ((torch.mean(per_ring_vals)[None] * ring_multipliers)
                * ctf_multipliers[:, None])
    assert_close(res, expected)
