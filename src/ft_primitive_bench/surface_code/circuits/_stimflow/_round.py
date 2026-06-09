"""Round-body builder shared by the chunks-backend primitives.

Mirrors :func:`ft_primitive_bench.surface_code.circuits._utils.append_surface_code_round`
but emits gates into a :class:`stimflow.ChunkBuilder` using complex-coordinate
targets. Detector emission is left to the caller via :meth:`ChunkBuilder.add_flow`.
"""

from __future__ import annotations

from typing import Tuple

import stimflow as sf

from .._tile import Patch
from .._utils import SCHEDULE_LAYER_MAPS


def _layer_cx_pairs(
    patch: Patch, layer: int, *, swap_xz_roles: bool = False
) -> list[Tuple[complex, complex]]:
    """Return (control, target) coordinate pairs for one CX layer."""
    pairs: list[Tuple[complex, complex]] = []
    for tile in patch.tiles:
        layer_map = SCHEDULE_LAYER_MAPS[tile.schedule_type]
        basis = tile.basis
        if swap_xz_roles:
            basis = "Z" if basis == "X" else "X"
        m = tile.measurement_qubit
        for direction, data_coord in tile.data_qubits.items():
            if data_coord is None:
                continue
            if layer_map[direction] != layer:
                continue
            if basis == "X":
                pairs.append((m, data_coord))
            else:
                pairs.append((data_coord, m))
    return pairs


def append_round_body(
    builder: sf.ChunkBuilder,
    patch: Patch,
    *,
    measure_key_prefix: str = "",
    swap_xz_roles: bool = False,
    measure_z_first: bool = False,
) -> None:
    """Append one CX-based syndrome-extraction round to `builder`.

    Measurement keys are ``(measure_key_prefix, ancilla_coord)``. Caller is
    responsible for ensuring ``measure_key_prefix`` is unique across chunks
    that share the same builder (none of our primitives do).

    Layer ordering and gate choice mirror
    :func:`_utils.append_surface_code_round` so that the resulting bit-level
    circuit matches the legacy backend exactly.
    """
    raw_x = sorted(
        (t.measurement_qubit for t in patch.tiles if t.basis == "X"),
        key=lambda c: (c.imag, c.real),
    )
    raw_z = sorted(
        (t.measurement_qubit for t in patch.tiles if t.basis == "Z"),
        key=lambda c: (c.imag, c.real),
    )
    # Legacy transversal_h swaps the meas_x/meas_z target lists when
    # swap_xz_roles is on. This swaps both the reset gate (RX vs R) and the
    # measurement gate (MX vs M) on each ancilla qubit — equivalent to
    # treating the originally-Z ancillas as X ancillas for one post-H round.
    if swap_xz_roles:
        x_meas_coords, z_meas_coords = raw_z, raw_x
    else:
        x_meas_coords, z_meas_coords = raw_x, raw_z

    # Determine layer count from any tile's schedule
    layer_count = 0
    for tile in patch.tiles:
        layer_map = SCHEDULE_LAYER_MAPS[tile.schedule_type]
        for direction, data_coord in tile.data_qubits.items():
            if data_coord is None:
                continue
            if layer_map[direction] + 1 > layer_count:
                layer_count = layer_map[direction] + 1

    init_ops = False
    if x_meas_coords:
        builder.append("RX", x_meas_coords)
        init_ops = True
    if z_meas_coords:
        builder.append("R", z_meas_coords)
        init_ops = True
    if init_ops:
        builder.append("TICK")

    for layer in range(layer_count):
        pairs = _layer_cx_pairs(patch, layer, swap_xz_roles=swap_xz_roles)
        if pairs:
            builder.append("CX", pairs)
            builder.append("TICK")

    key_func = (lambda q, p=measure_key_prefix: (p, q)) if measure_key_prefix else (lambda q: q)
    if measure_z_first:
        if z_meas_coords:
            builder.append("M", z_meas_coords, measure_key_func=key_func)
        if x_meas_coords:
            builder.append("MX", x_meas_coords, measure_key_func=key_func)
    else:
        if x_meas_coords:
            builder.append("MX", x_meas_coords, measure_key_func=key_func)
        if z_meas_coords:
            builder.append("M", z_meas_coords, measure_key_func=key_func)
