"""Tests for surface_code.circuits.S_gate."""
from __future__ import annotations

import stim

from ft_primitive_bench.surface_code.circuits import s_gate
from ft_primitive_bench.noise_models import uniform_depolarizing
from tests.conftest import assert_decodes_clean


def test_s_gate_builds():
    c = s_gate(
        distance=3, bridge_length=1, pre_rounds=1, merge_rounds=2,
        boundary_rounds=1, post_rounds=1,
    )
    assert isinstance(c, stim.Circuit)


def test_s_gate_dem_decomposes():
    clean = s_gate(
        distance=3, bridge_length=1, pre_rounds=1, merge_rounds=2,
        boundary_rounds=1, post_rounds=1,
    )
    noisy = uniform_depolarizing(p=1e-3).noisy_circuit(clean)
    assert_decodes_clean(noisy)
