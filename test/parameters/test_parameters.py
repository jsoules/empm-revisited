import torch
from inspect import get_annotations, getmembers
from types import MethodType
from unittest.mock import Mock, patch
from pytest import raises

from empm.parameters import Parameters


def test_all_members_are_annotated():
    a = Parameters()
    member_names = [x[0] for x in getmembers(a) if x[0][0:2] != '__']
    annotated_members = get_annotations(a).keys()
    for m in member_names:
        if type(getattr(a, m)) == MethodType: continue
        assert m in annotated_members


def test_check_equality_rejects_wrong_type_objects():
    a = Parameters()
    b = object()
    assert not a.check_equality(b) # type: ignore


def test_check_equality_rejects_on_differing_tensors():
    a = Parameters(r8_delta_x_requested_ = torch.ones(12) * 2)
    b = Parameters(r8_delta_x_requested_ = torch.ones(12))
    assert not a.check_equality(b)
    assert not b.check_equality(a)


def test_check_equality_rejects_on_differing_values():
    a = Parameters()
    b = Parameters()
    b.delta_r_max = a.delta_r_max + 1
    assert not a.check_equality(b)
    assert not b.check_equality(a)


def test_check_equality_rejects_on_missing_members():
    # NOTE: Since we're reading off the annotations, this will actually
    # fail the getattr, not the annotations count.
    a = Parameters()
    b = Parameters()
    del b.l_max
    assert not a.check_equality(b)
    assert not b.check_equality(a)


def test_check_equality_passes_on_match():
    a = Parameters()
    b = Parameters()
    assert a.check_equality(b)
    assert b.check_equality(a)


def test_dictionary_representation():
    orig = Parameters(
        flag_verbose = 3,
        n_iteration = 12,
        r8_delta_x_requested_ = torch.ones(12),
        r8_delta_y_requested_ = torch.ones(15),
        flag_p_vs_c = -4,
        n_a_degree = 128,
        tolerance_master = 19,
        tolerance_cluster = -1
    )

    round_trip = Parameters.from_dict(orig.to_dict())
    # raise ValueError(round_trip.to_dict())
    assert orig.check_equality(round_trip)


def test_set_empirical_ctf_rank():
    a = Parameters()
    mock_rank = 12
    mock_ctf = Mock()
    mock_ctf.empirically_determine_rank = Mock(return_value=mock_rank)

    assert a.rank_CTF < 0
    a.set_empirical_ctf_rank(mock_ctf, Mock(), Mock())
    assert a.rank_CTF == mock_rank
    mock_ctf.empirically_determine_rank.assert_called_once()


def test_set_empirical_ctf_rank_errors_if_already_set():
    a = Parameters(rank_CTF=12)
    with raises(Exception, match="manually set"):
        a.set_empirical_ctf_rank(Mock(), Mock(), Mock())


def test_verbosity_management():
    a = Parameters(flag_verbose=0)
    assert a.flag_verbose == 0
    a.reduce_verbosity()
    assert a.flag_verbose == 0

    b = Parameters(flag_verbose=2)
    b.reduce_verbosity()
    assert b.flag_verbose == 1
    b.restore_verbosity()
    assert b.flag_verbose == 2

