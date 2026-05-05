"""Shared utilities for surface-code circuit generation."""

from typing import Dict, List, Optional, Tuple

import stim
from ._tile import Patch


def c2xy(z: complex) -> Tuple[float, float]:
    """Convert complex x + y*1j into (x, y) coords."""
    return z.real, z.imag


def coord_sort_key(coord: complex) -> Tuple[float, float]:
    """Canonical (y, x) sort key for surface-code coordinate sets.

    Used by every primitive builder to produce a row-major qubit ordering.
    Rec-offset arithmetic in the builders depends on this ordering being
    consistent across `index_of` assignment, the `MX`/`MZ` target lists,
    and the per-coord position dicts (`merged_z_pos`, `ancilla_pos`, …).
    Do not introduce a second sort key in the same builder.
    """
    return (coord.imag, coord.real)


SCHEDULE_LAYER_MAPS: Dict[str, Dict[str, int]] = {
    "X": {"UR": 0, "DR": 1, "UL": 2, "DL": 3},   # regular X tile
    "Z": {"UR": 0, "UL": 1, "DR": 2, "DL": 3},   # regular Z tile
    "XX": {"DR": 0, "DL": 1, "UR": 2, "UL": 3},  # X-boundary surgery X tile
    "ZZ": {"DR": 0, "UR": 1, "DL": 2, "UL": 3},  # X-boundary surgery Z tile
}


def build_schedule(
    patch: Patch,
) -> Tuple[List[Tuple[complex, List[Tuple[int, str, complex]]]], int]:
    """Build a per-tile schedule of (meas_coord, [(layer_idx, basis, data_coord), ...])."""
    schedule: List[Tuple[complex, List[Tuple[int, str, complex]]]] = []
    max_layer = 0
    for tile in patch.tiles:
        layer_map = SCHEDULE_LAYER_MAPS[tile.schedule_type]
        tile_ops: List[Tuple[int, str, complex]] = []
        for direction, data_coord in tile.data_qubits.items():
            if data_coord is None:
                continue
            layer = layer_map[direction]
            tile_ops.append((layer, tile.basis, data_coord))
            if layer + 1 > max_layer:
                max_layer = layer + 1
        schedule.append((tile.measurement_qubit, tile_ops))
    return schedule, max_layer


def resolve_schedule(
    schedule: List[Tuple[complex, List[Tuple[int, str, complex]]]],
    index_of: Dict[complex, int],
) -> List[Tuple[int, List[Tuple[int, str, int]]]]:
    """Pre-resolve coordinate-to-index lookups in a schedule."""
    return [
        (
            index_of[meas_coord],
            [
                (direction_idx, basis_char, index_of[data_coord])
                for direction_idx, basis_char, data_coord in entries
            ],
        )
        for meas_coord, entries in schedule
    ]


def append_surface_code_round(
    circuit: stim.Circuit,
    resolved_sched: List[Tuple[int, List[Tuple[int, str, int]]]],
    layer_count: int,
    meas_x: List[int],
    meas_z: List[int],
    *,
    swap_xz_roles: bool = False,
    measure_z_first: bool = False,
    end_tick: bool = True,
) -> None:
    """Append one CX-based syndrome-extraction round.

    X checks use the ancilla as the CNOT control; Z checks use the data
    qubit as the control. ``swap_xz_roles`` flips this for the post-H
    rounds in ``transversal_h``.
    """
    def effective_basis(raw_basis: str) -> str:
        if not swap_xz_roles:
            return raw_basis
        return "Z" if raw_basis == "X" else "X"

    init_ops = False
    if meas_x:
        circuit.append("RX", meas_x)
        init_ops = True
    if meas_z:
        circuit.append("RZ", meas_z)
        init_ops = True
    if init_ops:
        circuit.append("TICK")

    for layer in range(layer_count):
        cx_targets: List[int] = []
        for mq_idx, ops in resolved_sched:
            for direction_idx, basis_char, dq_idx in ops:
                if direction_idx != layer:
                    continue
                basis = effective_basis(basis_char)
                if basis == "X":
                    cx_targets += [mq_idx, dq_idx]
                elif basis == "Z":
                    cx_targets += [dq_idx, mq_idx]
        if cx_targets:
            circuit.append("CX", cx_targets)
            circuit.append("TICK")

    if measure_z_first:
        if meas_z:
            circuit.append("MZ", meas_z)
        if meas_x:
            circuit.append("MX", meas_x)
    else:
        if meas_x:
            circuit.append("MX", meas_x)
        if meas_z:
            circuit.append("MZ", meas_z)
    if (meas_x or meas_z) and end_tick:
        circuit.append("TICK")


def build_surface_code_round_circuit(
    resolved_sched: List[Tuple[int, List[Tuple[int, str, int]]]],
    layer_count: int,
    meas_x: List[int],
    meas_z: List[int],
    *,
    swap_xz_roles: bool = False,
    measure_z_first: bool = False,
    end_tick: bool = True,
) -> stim.Circuit:
    """Build one complete CX-based syndrome-extraction round as a standalone fragment."""
    circuit = stim.Circuit()
    append_surface_code_round(
        circuit,
        resolved_sched,
        layer_count,
        meas_x,
        meas_z,
        swap_xz_roles=swap_xz_roles,
        measure_z_first=measure_z_first,
        end_tick=end_tick,
    )
    return circuit


def append_repeated_phase(
    circuit: stim.Circuit,
    count: int,
    first_round: stim.Circuit,
    *,
    middle_round: Optional[stim.Circuit] = None,
    last_round: Optional[stim.Circuit] = None,
) -> None:
    """Append a phase using explicit first/last rounds and a repeated middle."""
    if count <= 0:
        return

    middle = first_round if middle_round is None else middle_round
    last = first_round if last_round is None else last_round

    circuit += first_round
    if count == 1:
        return
    if count == 2:
        circuit += last
        return
    circuit += middle * (count - 2)
    circuit += last
