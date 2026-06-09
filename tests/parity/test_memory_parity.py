"""Parity tests: chunks backend produces equivalent circuits to legacy."""

from __future__ import annotations

import pytest

from ft_primitive_bench.surface_code.circuits import memory

from ._harness import assert_dem_equivalent


@pytest.mark.parametrize("x_distance,z_distance", [(3, 3), (3, 5), (5, 3), (5, 5)])
@pytest.mark.parametrize("rounds", [1, 2, 3, 5])
@pytest.mark.parametrize("meas_basis", ["Z", "X"])
def test_memory_chunks_matches_legacy(x_distance, z_distance, rounds, meas_basis):
    legacy = memory(x_distance, z_distance, rounds, meas_basis=meas_basis, backend="legacy")
    chunks = memory(x_distance, z_distance, rounds, meas_basis=meas_basis, backend="stimflow")
    assert_dem_equivalent(legacy, chunks)


@pytest.mark.parametrize("x_distance,z_distance", [(3, 3), (5, 5)])
def test_memory_chunks_verify(x_distance, z_distance):
    """Every chunk's declared flows are validated by the circuit it builds."""
    from ft_primitive_bench.surface_code.circuits._stimflow.memory import build_memory_chunks

    chunks = build_memory_chunks(x_distance, z_distance, rounds=3, meas_basis="Z")
    for chunk in chunks:
        chunk.verify()
