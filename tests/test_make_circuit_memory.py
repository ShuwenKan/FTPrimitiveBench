"""Tests for surface_code.circuits._make_circuit.memory."""

from __future__ import annotations

import pytest
import stim

from ft_primitive_bench.surface_code.circuits import memory
from tests.conftest import assert_decodes_clean


def test_memory_z_basis_basic():
    c = memory(x_distance=3, z_distance=3, rounds=3, meas_basis="Z")
    assert isinstance(c, stim.Circuit)
    assert "DETECTOR" in str(c)
    assert "OBSERVABLE_INCLUDE" in str(c)


def test_memory_x_basis_basic():
    c = memory(x_distance=3, z_distance=3, rounds=3, meas_basis="X")
    assert isinstance(c, stim.Circuit)


def test_memory_rectangular():
    c = memory(x_distance=3, z_distance=5, rounds=2, meas_basis="Z")
    assert isinstance(c, stim.Circuit)


def test_memory_rounds_below_one_raises():
    with pytest.raises(ValueError):
        memory(x_distance=3, z_distance=3, rounds=0, meas_basis="Z")


def test_memory_dem_decomposes():
    from ft_primitive_bench.noise_models import uniform_depolarizing

    clean = memory(x_distance=3, z_distance=3, rounds=3, meas_basis="Z")
    noisy = uniform_depolarizing(p=1e-3).noisy_circuit(clean)
    assert_decodes_clean(noisy)


def test_memory_invalid_basis_raises():
    with pytest.raises((ValueError, KeyError)):
        memory(x_distance=3, z_distance=3, rounds=2, meas_basis="Q")
