"""Parity tests for the chunks transversal_h primitive."""

from __future__ import annotations

import pytest

from ft_primitive_bench.surface_code.circuits import transversal_h

from ._harness import assert_dem_equivalent


@pytest.mark.parametrize("x_distance,z_distance", [(3, 3), (3, 5), (5, 3), (5, 5)])
@pytest.mark.parametrize("pre_rounds,post_rounds", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 3)])
@pytest.mark.parametrize("meas_basis", ["Z", "X"])
def test_transversal_h_chunks_matches_legacy(
    x_distance, z_distance, pre_rounds, post_rounds, meas_basis
):
    legacy = transversal_h(
        x_distance, z_distance, pre_rounds, post_rounds, meas_basis=meas_basis, backend="legacy"
    )
    chunks = transversal_h(
        x_distance, z_distance, pre_rounds, post_rounds, meas_basis=meas_basis, backend="stimflow"
    )
    assert_dem_equivalent(legacy, chunks)


@pytest.mark.parametrize("x_distance,z_distance", [(3, 3), (5, 5)])
def test_transversal_h_chunks_verify(x_distance, z_distance):
    from ft_primitive_bench.surface_code.circuits._stimflow.transversal_h import (
        build_transversal_h_chunks,
    )

    chunks = build_transversal_h_chunks(
        x_distance, z_distance, pre_rounds=1, post_rounds=1, meas_basis="Z"
    )
    for chunk in chunks:
        chunk.verify()
