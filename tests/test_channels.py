"""Tests for surface_code.noise_models.channels."""
from __future__ import annotations

import pytest

from ft_primitive_bench.noise_models.channels import (
    MeasureFlip,
    PauliChannel1,
    PauliChannel2,
    ResetFlip,
    clip_probability,
    make_1q_biased,
    make_1q_depolarizing,
    make_2q_biased,
    make_2q_depolarizing,
    make_measurement_flip_channel,
    make_reset_flip_channel,
)


def test_clip_probability_clamps():
    assert clip_probability(-0.5) == 0.0
    assert clip_probability(0.5) == 0.5
    assert clip_probability(1.5) == 1.0


def test_pauli1_depolarizing_uniform():
    ch = PauliChannel1.depolarizing(0.03)
    assert ch.px == pytest.approx(0.01)
    assert ch.py == pytest.approx(0.01)
    assert ch.pz == pytest.approx(0.01)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_pauli1_biased_normalized_to_p(axis):
    p = 0.03
    eta = 5.0
    ch = PauliChannel1.biased(p, axis, eta)
    assert ch.px + ch.py + ch.pz == pytest.approx(p)


def test_pauli1_biased_eta_one_recovers_depolarizing():
    p = 0.03
    ch = PauliChannel1.biased(p, "Z", 1.0)
    dep = PauliChannel1.depolarizing(p)
    assert ch == dep


def test_pauli1_biased_invalid_axis():
    with pytest.raises(ValueError):
        PauliChannel1.biased(0.01, "W", 5.0)


def test_pauli2_depolarizing_total_p():
    p = 0.03
    ch = PauliChannel2.depolarizing(p)
    assert sum(ch.probs) == pytest.approx(p)
    assert len(ch.probs) == 15


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_pauli2_biased_normalized(axis):
    p = 0.03
    ch = PauliChannel2.biased(p, axis, 5.0)
    assert sum(ch.probs) == pytest.approx(p)


def test_make_1q_biased_eta_one_collapses_to_depolarize1():
    name, val = make_1q_biased(0.01, "Z", 1.0)
    assert name == "DEPOLARIZE1"


def test_make_1q_biased_eta_high_returns_pauli_channel_1():
    name, val = make_1q_biased(0.01, "Z", 5.0)
    assert name == "PAULI_CHANNEL_1"
    assert isinstance(val, list) and len(val) == 3


def test_make_2q_biased_eta_one_collapses_to_depolarize2():
    name, val = make_2q_biased(0.01, "Z", 1.0)
    assert name == "DEPOLARIZE2"


def test_make_2q_biased_eta_high_returns_pauli_channel_2():
    name, val = make_2q_biased(0.01, "Z", 5.0)
    assert name == "PAULI_CHANNEL_2"
    assert isinstance(val, list) and len(val) == 15


def test_make_1q_depolarizing_basic():
    name, val = make_1q_depolarizing(0.01)
    assert name == "DEPOLARIZE1"
    assert val == pytest.approx(0.01)


def test_make_2q_depolarizing_basic():
    name, val = make_2q_depolarizing(0.02)
    assert name == "DEPOLARIZE2"
    assert val == pytest.approx(0.02)


def test_measurement_flip_z_basis():
    z, x = make_measurement_flip_channel(0.01, basis="Z")
    assert z == pytest.approx(0.01)
    assert x == 0.0


def test_measurement_flip_x_basis():
    z, x = make_measurement_flip_channel(0.01, basis="X")
    assert x == pytest.approx(0.01)
    assert z == 0.0


def test_measurement_flip_unspecified_basis():
    z, x = make_measurement_flip_channel(0.01)
    assert z == pytest.approx(0.01)
    assert x == pytest.approx(0.01)


def test_reset_flip_z_basis():
    z, x = make_reset_flip_channel(0.01, basis="Z")
    assert z == pytest.approx(0.01)
    assert x == 0.0


def test_reset_flip_unsupported_basis():
    with pytest.raises(ValueError):
        make_measurement_flip_channel(0.01, basis="W")


def test_pauli1_immutable():
    ch = PauliChannel1.depolarizing(0.01)
    with pytest.raises(Exception):
        ch.px = 1.0  # type: ignore[misc]


def test_measure_flip_dataclass():
    mf = MeasureFlip(z_flip=0.01, x_flip=0.02)
    assert mf.to_spec() == (pytest.approx(0.01), pytest.approx(0.02))


def test_reset_flip_dataclass():
    rf = ResetFlip(z_reset_flip=0.01, x_reset_flip=0.02)
    assert rf.to_spec() == (pytest.approx(0.01), pytest.approx(0.02))
