"""Parity tests for the chunks s_gate primitive.

The stimflow backend currently delegates to legacy (see _stimflow/s_gate.py),
so this parity test is trivially satisfied by an exact-circuit-equality
check rather than a DEM canonicalization. When the chunks port is
upgraded to a real stimflow implementation, switch back to
assert_dem_equivalent.
"""

from __future__ import annotations

import pytest

from ft_primitive_bench.surface_code.circuits.S_gate import s_gate


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize(
    "bridge_length,pre_rounds,merge_rounds,boundary_rounds,post_rounds",
    [
        (1, 1, 1, 1, 1),
        (1, 1, 3, 2, 1),
        (3, 1, 1, 1, 1),
        (3, 2, 2, 2, 1),
    ],
)
def test_s_gate_chunks_matches_legacy(
    distance, bridge_length, pre_rounds, merge_rounds, boundary_rounds, post_rounds
):
    if (distance + bridge_length) % 2 != 0:
        pytest.skip("Invalid (distance + bridge_length) parity for ZZ surgery.")
    legacy = s_gate(
        distance=distance,
        bridge_length=bridge_length,
        pre_rounds=pre_rounds,
        merge_rounds=merge_rounds,
        boundary_rounds=boundary_rounds,
        post_rounds=post_rounds,
        backend="legacy",
    )
    chunks = s_gate(
        distance=distance,
        bridge_length=bridge_length,
        pre_rounds=pre_rounds,
        merge_rounds=merge_rounds,
        boundary_rounds=boundary_rounds,
        post_rounds=post_rounds,
        backend="stimflow",
    )
    # Currently identical via delegation; future chunks port should still
    # pass assert_dem_equivalent.
    assert legacy == chunks
