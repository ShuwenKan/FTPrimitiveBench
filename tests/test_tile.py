"""Tests for surface_code.circuits._tile."""
from __future__ import annotations

import pytest

from ft_primitive_bench.surface_code.circuits._tile import Patch, Tile


def _make_tile(basis="Z", m=1.5 + 1.5j) -> Tile:
    return Tile(
        basis=basis,
        measurement_qubit=m,
        data_qubits={
            "UL": m + (-0.5 - 0.5j),
            "UR": m + (0.5 - 0.5j),
            "DL": m + (-0.5 + 0.5j),
            "DR": m + (0.5 + 0.5j),
        },
    )


def test_tile_basic_construction():
    tile = _make_tile()
    assert tile.basis == "Z"
    assert tile.schedule_type == "Z"
    assert sum(v is not None for v in tile.data_qubits.values()) == 4


def test_tile_invalid_basis_raises():
    with pytest.raises(ValueError):
        _make_tile(basis="Y")


def test_tile_invalid_data_key_raises():
    with pytest.raises(ValueError):
        Tile(basis="Z", measurement_qubit=1 + 1j, data_qubits={"FOO": 0 + 0j})


def test_tile_non_complex_measurement_raises():
    with pytest.raises(TypeError):
        Tile(basis="Z", measurement_qubit=(1.0, 1.0), data_qubits={})  # type: ignore[arg-type]


def test_patch_append_and_iterate():
    p = Patch()
    p.append(_make_tile(m=1.5 + 1.5j))
    p.append(_make_tile(m=2.5 + 1.5j))
    assert len(p) == 2
    assert len(list(p)) == 2


def test_patch_replaces_duplicate_measurement_coord():
    p = Patch()
    p.append(_make_tile(basis="Z", m=1.5 + 1.5j))
    p.append(_make_tile(basis="X", m=1.5 + 1.5j))
    assert len(p) == 1
    assert p[0].basis == "X"


def test_patch_extend_batch():
    p = Patch()
    p.extend([_make_tile(m=1.5 + 1.5j), _make_tile(m=2.5 + 1.5j)])
    assert len(p) == 2
