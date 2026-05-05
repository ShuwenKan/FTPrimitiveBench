"""Tests for surface_code.circuits._make_circuit.transversal_h."""

from __future__ import annotations

import pytest
import stim

from ft_primitive_bench.noise_models import uniform_depolarizing
from ft_primitive_bench.surface_code.circuits import transversal_h
from tests.conftest import assert_decodes_clean


@pytest.mark.parametrize("meas_basis", ["Z", "X"])
def test_transversal_h_builds(meas_basis):
    c = transversal_h(
        x_distance=3, z_distance=3, pre_rounds=2, post_rounds=2, meas_basis=meas_basis
    )
    assert isinstance(c, stim.Circuit)
    assert "H " in str(c) or "H_XZ" in str(c) or "\nH" in str(c)


def test_transversal_h_dem_decomposes():
    clean = transversal_h(x_distance=3, z_distance=3, pre_rounds=2, post_rounds=2, meas_basis="Z")
    noisy = uniform_depolarizing(p=1e-3).noisy_circuit(clean)
    assert_decodes_clean(noisy)


def test_transversal_h_zero_pre_rounds():
    c = transversal_h(x_distance=3, z_distance=3, pre_rounds=0, post_rounds=2, meas_basis="Z")
    assert isinstance(c, stim.Circuit)


def test_transversal_h_zero_post_rounds():
    c = transversal_h(x_distance=3, z_distance=3, pre_rounds=2, post_rounds=0, meas_basis="Z")
    assert isinstance(c, stim.Circuit)
