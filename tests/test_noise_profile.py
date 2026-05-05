"""Tests for surface_code.noise_models.noise_profile."""

from __future__ import annotations

import pytest

from ft_primitive_bench.noise_models import Coherence, NoiseProfile


def test_global_entry_basic():
    profile = NoiseProfile()
    profile[None, None] = {"p_1q": 1e-3, "p_meas": 5e-3}
    assert profile[None, None] == {"p_1q": 1e-3, "p_meas": 5e-3}


def test_qubit_override_entry():
    profile = NoiseProfile()
    profile[5, None] = {"p_meas": 1e-2}
    assert profile[5, None]["p_meas"] == 1e-2


def test_pair_override_entry():
    profile = NoiseProfile()
    profile[(3, 4), None] = {"p_2q": 2e-3}
    assert profile[(3, 4), None]["p_2q"] == 2e-3


def test_round_only_override_entry():
    profile = NoiseProfile()
    profile[None, 7] = {"p_1q": 2e-3}


def test_qubit_round_override_entry():
    profile = NoiseProfile()
    profile[5, 7] = {"p_meas": 5e-2}


def test_pair_keyed_qubit_field_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[(3, 4), None] = {"p_meas": 1e-2}  # p_meas not pair-local


def test_qubit_keyed_p_2q_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[5, None] = {"p_2q": 1e-3}


def test_unknown_field_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, None] = {"p_bogus": 1e-3}


def test_t1_without_t2_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, None] = {"T1": 30e-6}


def test_mode_b_with_p_idle_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, None] = {"T1": 30e-6, "T2": 20e-6, "p_idle": 1e-4}


def test_p_idle_meas_without_p_idle_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, None] = {"p_idle_meas": 1e-3}


def test_invalid_key_shape_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[5] = {"p_meas": 1e-3}  # type: ignore[index]


def test_negative_qubit_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[-1, None] = {"p_1q": 1e-3}


def test_rate_above_one_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, None] = {"p_1q": 1.5}


def test_rate_below_zero_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, None] = {"p_1q": -0.1}


def test_round_negative_raises():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, -1] = {"p_1q": 1e-3}


def test_rate_duration_tuple():
    profile = NoiseProfile()
    profile[None, None] = {"T1": 30e-6, "T2": 20e-6, "p_1q": (1e-4, 40e-9)}
    assert profile[None, None]["p_1q"] == (1e-4, 40e-9)


def test_rate_duration_tuple_negative_duration():
    profile = NoiseProfile()
    with pytest.raises(ValueError):
        profile[None, None] = {"p_1q": (1e-4, -1e-9)}


def test_qubits_pairs_rounds_methods():
    profile = NoiseProfile()
    profile[None, None] = {"p_1q": 1e-3}
    profile[5, None] = {"p_meas": 1e-2}
    profile[(3, 4), None] = {"p_2q": 2e-3}
    profile[None, 7] = {"p_1q": 2e-3}
    profile[2, 7] = {"p_meas": 5e-2}
    assert 5 in profile.qubits()
    assert 2 in profile.qubits()
    assert (3, 4) in profile.pairs()
    assert 7 in profile.rounds()


def test_has_round_overrides():
    profile = NoiseProfile()
    profile[None, None] = {"p_1q": 1e-3}
    assert profile.has_round_overrides() is False
    profile[5, 7] = {"p_meas": 1e-2}
    assert profile.has_round_overrides() is True


def test_initial_dict_argument():
    profile = NoiseProfile({(None, None): {"p_1q": 1e-3}})
    assert profile[None, None] == {"p_1q": 1e-3}


def test_update_method_validates():
    profile = NoiseProfile()
    profile.update({(None, None): {"p_1q": 1e-3}})
    assert profile[None, None] == {"p_1q": 1e-3}
    with pytest.raises(ValueError):
        profile.update({(None, None): {"p_bogus": 1e-3}})


def test_coherence_dataclass():
    c = Coherence(T1=30e-6, T2=20e-6)
    assert c.T1 == 30e-6
    assert c.T2 == 20e-6


def test_coherence_negative_raises():
    with pytest.raises(ValueError):
        Coherence(T1=-1, T2=20e-6)


def test_coherence_immutable():
    c = Coherence(T1=30e-6, T2=20e-6)
    with pytest.raises(Exception):
        c.T1 = 1.0  # type: ignore[misc]
