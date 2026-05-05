"""Tests for surface_code.circuits._utils."""

from __future__ import annotations

import stim

from ft_primitive_bench.surface_code.circuits import rectangular_surface_code_patch
from ft_primitive_bench.surface_code.circuits._utils import (
    append_repeated_phase,
    build_schedule,
    c2xy,
    coord_sort_key,
    resolve_schedule,
)


def test_c2xy_basic():
    assert c2xy(1.5 + 2.5j) == (1.5, 2.5)


def test_coord_sort_key_orders_row_major():
    coords = [2 + 1j, 1 + 1j, 1 + 2j, 2 + 2j]
    ordered = sorted(coords, key=coord_sort_key)
    # row-major: y first, then x
    assert ordered == [1 + 1j, 2 + 1j, 1 + 2j, 2 + 2j]


def test_build_schedule_returns_max_layer_4():
    patch = rectangular_surface_code_patch(x_distance=3, z_distance=3)
    schedule, max_layer = build_schedule(patch)
    assert max_layer == 4
    assert len(schedule) == len(patch.tiles)


def test_resolve_schedule_remaps_indices():
    patch = rectangular_surface_code_patch(x_distance=3, z_distance=3)
    schedule, _ = build_schedule(patch)
    coords = sorted(patch.data_set | patch.measure_set, key=coord_sort_key)
    index_of = {c: i for i, c in enumerate(coords)}
    resolved = resolve_schedule(schedule, index_of)
    assert len(resolved) == len(schedule)
    for ancilla_idx, ops in resolved:
        assert isinstance(ancilla_idx, int)
        for direction_idx, basis_char, data_idx in ops:
            assert basis_char in {"X", "Z"}
            assert isinstance(data_idx, int)


def test_append_repeated_phase_count_zero_noop():
    circ = stim.Circuit()
    fragment = stim.Circuit("H 0")
    append_repeated_phase(circ, count=0, first_round=fragment)
    assert len(circ) == 0


def test_append_repeated_phase_count_one():
    circ = stim.Circuit()
    fragment = stim.Circuit("H 0")
    append_repeated_phase(circ, count=1, first_round=fragment)
    assert len(circ) == 1


def test_append_repeated_phase_count_three_uses_middle():
    circ = stim.Circuit()
    first = stim.Circuit("H 0\nTICK")
    middle = stim.Circuit("S 1\nTICK")
    last = stim.Circuit("X 2\nTICK")
    append_repeated_phase(circ, count=3, first_round=first, middle_round=middle, last_round=last)
    s = str(circ)
    assert "H 0" in s
    assert "S 1" in s
    assert "X 2" in s
