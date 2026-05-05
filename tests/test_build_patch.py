"""Tests for surface_code.circuits._build_patch."""

from __future__ import annotations

import pytest

from ft_primitive_bench.surface_code.circuits import (
    boundary_tiles,
    checkerboard_weight4_patch,
    create_individual_patch,
    rectangular_surface_code_patch,
)


def test_rectangular_patch_3x3():
    patch = rectangular_surface_code_patch(x_distance=3, z_distance=3)
    assert len(patch.data_set) == 9


def test_rectangular_patch_3x5():
    patch = rectangular_surface_code_patch(x_distance=3, z_distance=5)
    assert len(patch.data_set) == 15


def test_rectangular_patch_swapped_basis():
    patch = rectangular_surface_code_patch(
        x_distance=3,
        z_distance=3,
        top_bottom_basis="X",
        left_right_basis="Z",
    )
    assert len(patch.data_set) == 9
    assert len(patch.measure_set) > 0


def test_rectangular_patch_invalid_basis_raises():
    with pytest.raises(ValueError):
        rectangular_surface_code_patch(x_distance=3, z_distance=3, top_bottom_basis="Y")


def test_rectangular_patch_invalid_distance_raises():
    with pytest.raises(ValueError):
        rectangular_surface_code_patch(x_distance=0, z_distance=3)


def test_checkerboard_weight4_patch_basic():
    patch = checkerboard_weight4_patch(x_distance=3, z_distance=3)
    assert len(patch.tiles) == 9
    for tile in patch.tiles:
        assert sum(v is not None for v in tile.data_qubits.values()) == 4


def test_create_individual_patch_two_separated():
    patch = create_individual_patch(x_distance=3, z_distance=3, bridge_length=1)
    assert len(patch.data_set) >= 2 * 9


def test_boundary_tiles_basic():
    patch = boundary_tiles(orientation="top", length=2, basis="Z", start=0.5 + 0.5j)
    assert len(patch.tiles) == 2
    for tile in patch.tiles:
        assert tile.basis == "Z"


def test_boundary_tiles_invalid_basis():
    with pytest.raises(ValueError):
        boundary_tiles(orientation="top", length=2, basis="Y", start=0.5 + 0.5j)


def test_boundary_tiles_invalid_orientation():
    with pytest.raises(ValueError):
        boundary_tiles(orientation="diagonal", length=2, basis="Z", start=0.5 + 0.5j)


def test_boundary_tiles_zero_length_raises():
    with pytest.raises(ValueError):
        boundary_tiles(orientation="top", length=0, basis="Z", start=0.5 + 0.5j)
