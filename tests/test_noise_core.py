"""Tests for surface_code.noise_models._noise_core (private but load-bearing)."""
from __future__ import annotations

import math

import pytest

from ft_primitive_bench.noise_models import idle_pauli_channel_from_T1T2


def test_idle_pauli_channel_returns_three_probs():
    px, py, pz = idle_pauli_channel_from_T1T2(t=100e-9, T1=30e-6, T2=20e-6)
    assert px >= 0 and py >= 0 and pz >= 0


def test_idle_pauli_channel_zero_duration():
    px, py, pz = idle_pauli_channel_from_T1T2(t=0.0, T1=30e-6, T2=20e-6)
    assert px == pytest.approx(0.0)
    assert py == pytest.approx(0.0)
    assert pz == pytest.approx(0.0)


def test_idle_pauli_channel_long_duration_saturates():
    # As t -> infinity, T1-driven px = py -> 0.25, total -> 0.75 (max depolarizing).
    px, py, pz = idle_pauli_channel_from_T1T2(t=1.0, T1=30e-6, T2=20e-6)
    assert px == pytest.approx(0.25, abs=1e-6)
    assert py == pytest.approx(0.25, abs=1e-6)


def test_idle_pauli_channel_px_equals_py():
    px, py, _ = idle_pauli_channel_from_T1T2(t=100e-9, T1=30e-6, T2=20e-6)
    assert px == py


def test_idle_pauli_channel_t2_equal_2t1_yields_no_pure_z():
    # For T2 = 2*T1 the dephasing matches the T1-driven pz, leaving zero pure pz.
    T1 = 30e-6
    T2 = 2 * T1
    _, _, pz = idle_pauli_channel_from_T1T2(t=1e-7, T1=T1, T2=T2)
    assert pz == pytest.approx(0.0, abs=1e-5)
