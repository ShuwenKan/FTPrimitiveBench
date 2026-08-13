"""Parity tests for the chunks lattice_surgery primitive."""

from __future__ import annotations

import pytest

from ft_primitive_bench.surface_code.circuits import lattice_surgery

from ._harness import assert_dem_equivalent


# Note: x_distance + bridge_length must be even for MXX; z_distance + bridge_length
# must be even for MZZ. See legacy parity check at _make_circuit.py:241-247.
@pytest.mark.parametrize(
    "x,z,bridge,meas",
    [
        (3, 3, 1, "Z"),
        (3, 3, 1, "X"),
        (3, 3, 3, "Z"),
        (5, 5, 1, "Z"),
        (5, 5, 1, "X"),
    ],
)
@pytest.mark.parametrize("pre,merge,post", [(0, 1, 0), (1, 1, 1), (2, 3, 2)])
def test_lattice_surgery_chunks_matches_legacy(x, z, bridge, meas, pre, merge, post):
    legacy = lattice_surgery(
        x, z, bridge, pre, merge, post, meas_basis=meas, backend="legacy"
    )
    chunks = lattice_surgery(
        x, z, bridge, pre, merge, post, meas_basis=meas, backend="stimflow"
    )
    assert_dem_equivalent(legacy, chunks)


def test_lattice_surgery_init_chunk_verify():
    """The init chunk's flows are deterministic from data reset — must verify cleanly.

    Round / merge / destruct chunks have L = L1*L2 flows that anticommute with
    individual X-stab measurements within one round (L1*L2 only becomes a
    deterministic observable across the whole sequence after error correction).
    Per-chunk verify fails for those, but the compiled circuit's flow chain is
    correct, as confirmed by the DEM-parity sweep above.
    """
    from ft_primitive_bench.surface_code.circuits._stimflow.lattice_surgery import (
        build_lattice_surgery_chunks,
    )

    chunks = build_lattice_surgery_chunks(
        x_distance=3, z_distance=3, bridge_length=1, pre_rounds=1,
        merge_rounds=2, post_rounds=1, meas_basis="Z",
    )
    # Verify just the init chunk; the rest depend on cross-chunk flow context.
    chunks[0].verify()
