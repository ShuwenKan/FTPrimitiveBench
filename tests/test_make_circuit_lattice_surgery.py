"""Tests for surface_code.circuits._make_circuit.lattice_surgery + helpers."""

from __future__ import annotations

import pytest
import stim

from ft_primitive_bench.noise_models import uniform_depolarizing
from ft_primitive_bench.surface_code.circuits import (
    build_merged_patch,
    lattice_surgery,
    patch_midlines,
)
from tests.conftest import assert_decodes_clean


def test_lattice_surgery_z_basis_basic():
    c = lattice_surgery(
        x_distance=3,
        z_distance=3,
        bridge_length=1,
        pre_rounds=1,
        merge_rounds=2,
        post_rounds=1,
        meas_basis="Z",
    )
    assert isinstance(c, stim.Circuit)


def test_lattice_surgery_x_basis_basic():
    c = lattice_surgery(
        x_distance=3,
        z_distance=3,
        bridge_length=1,
        pre_rounds=1,
        merge_rounds=2,
        post_rounds=1,
        meas_basis="X",
    )
    assert isinstance(c, stim.Circuit)


def test_lattice_surgery_dem_decomposes():
    clean = lattice_surgery(
        x_distance=3,
        z_distance=3,
        bridge_length=1,
        pre_rounds=1,
        merge_rounds=2,
        post_rounds=1,
        meas_basis="Z",
    )
    noisy = uniform_depolarizing(p=1e-3).noisy_circuit(clean)
    assert_decodes_clean(noisy)


def test_lattice_surgery_z_parity_constraint_violated_raises():
    # z_distance=3, bridge=2 -> sum=5 odd -> should raise
    with pytest.raises(ValueError):
        lattice_surgery(
            x_distance=3,
            z_distance=3,
            bridge_length=2,
            pre_rounds=1,
            merge_rounds=2,
            post_rounds=1,
            meas_basis="Z",
        )


def test_lattice_surgery_x_parity_constraint_violated_raises():
    # x_distance=3, bridge=2 -> sum=5 odd -> should raise
    with pytest.raises(ValueError):
        lattice_surgery(
            x_distance=3,
            z_distance=3,
            bridge_length=2,
            pre_rounds=1,
            merge_rounds=2,
            post_rounds=1,
            meas_basis="X",
        )


def test_build_merged_patch_returns_two_patches():
    individual, merged = build_merged_patch(
        x_distance=3,
        z_distance=3,
        bridge_length=1,
        horizontal=False,
    )
    assert individual is not None
    assert merged is not None
    assert len(merged.measure_set) > len(individual.measure_set)


def test_patch_midlines_horizontal_and_vertical():
    individual, _ = build_merged_patch(
        x_distance=3,
        z_distance=3,
        bridge_length=1,
        horizontal=False,
    )
    h_lo, h_hi = patch_midlines(individual, horizontal=True)
    v_lo, v_hi = patch_midlines(individual, horizontal=False)
    assert h_lo <= h_hi
    assert v_lo <= v_hi
