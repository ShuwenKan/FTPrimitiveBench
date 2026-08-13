"""Chunks-backend memory primitive.

Decomposes the legacy ``memory`` primitive into four logical chunks:

    init_chunk             # R/RX on data; declares meas_basis stabilizer end-flows
    round_first_chunk      # full round; emit-only flows for opp-basis stabs
    round_chunk            # full round; passthrough for ALL stabs
    final_chunk            # M/MX on data; absorb meas_basis stabs into data parities

Composition::

    rounds == 1: [init, round_first, final]
    rounds  >= 2: [init, round_first, round_chunk * (rounds-1), final]

The round chunks differ in how they handle the opposite-basis stabilizers
during round 1: those stabilizers are random after data reset, so the legacy
backend does NOT emit a detector for them in round 1 but DOES from round 2
onward (comparing M(round_k) XOR M(round_k-1)). In the chunks framework, this
is encoded by having ``round_first_chunk`` emit-only the opp-basis flows
(end=stab, no start), so subsequent rounds can chain start=stab against them
and emit the 2-rec detector.
"""

from __future__ import annotations

from typing import List

import stim
import stimflow as sf

from .._build_patch import rectangular_surface_code_patch
from .._tile import Patch
from .._utils import c2xy, coord_sort_key
from ._round import append_round_body

_OBS_NAME = "L"


def _logical_strip_coords(patch: Patch, meas_basis: str) -> List[complex]:
    """Midline data strip in the measurement basis (matches legacy)."""
    data = sorted(patch.data_set, key=coord_sort_key)
    if not data:
        return []
    xs = [round(c2xy(q)[0]) for q in data]
    ys = [round(c2xy(q)[1]) for q in data]
    mid_x = (min(xs) + max(xs)) // 2
    mid_y = (min(ys) + max(ys)) // 2
    if meas_basis == "Z":
        return [q for q in data if round(c2xy(q)[0]) == mid_x]
    return [q for q in data if round(c2xy(q)[1]) == mid_y]


def _logical_pauli_map(patch: Patch, meas_basis: str) -> sf.PauliMap:
    strip = _logical_strip_coords(patch, meas_basis)
    return sf.PauliMap({q: meas_basis for q in strip}).with_obs_name(_OBS_NAME)


def _tile_pauli_map(tile) -> sf.PauliMap:
    return sf.PauliMap(
        {coord: tile.basis for coord in tile.data_qubits.values() if coord is not None}
    )


def build_init_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    """Data reset in meas_basis. End-flows for meas_basis stabs; discard the rest."""
    builder = sf.ChunkBuilder()
    data_coords = sorted(patch.data_set, key=coord_sort_key)
    init_gate = "R" if meas_basis == "Z" else "RX"
    builder.append(init_gate, data_coords)

    for tile in patch.tiles:
        pm = _tile_pauli_map(tile)
        if tile.basis == meas_basis:
            builder.add_flow(end=pm, center=tile.measurement_qubit)
        else:
            builder.add_discarded_flow_output(pm)

    builder.add_flow(end=_logical_pauli_map(patch, meas_basis))
    return builder.finish_chunk()


def build_round_first_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    """First round after init.

    Per the canonical stimflow idle-round pattern (see ChunkBuilder docstring),
    each stabilizer measurement is declared as TWO flows referencing the same
    ancilla measurement: an absorb (start=stab, m=[anc]) and an emit
    (end=stab, m=[anc]). Combined with the previous chunk's open emit flow,
    these produce one DETECTOR per stabilizer per round.

    The opposite-basis stabilizers are random after init's data reset, so init
    declares them as discarded outputs. In this first round we therefore omit
    the absorb flow for opp-basis tiles, only emitting them so round 2 can
    absorb them and start producing detectors.
    """
    builder = sf.ChunkBuilder()
    append_round_body(builder, patch)

    for tile in patch.tiles:
        pm = _tile_pauli_map(tile)
        ancilla_key = tile.measurement_qubit
        if tile.basis == meas_basis:
            builder.add_flow(start=pm, measurements=[ancilla_key], center=tile.measurement_qubit)
        builder.add_flow(end=pm, measurements=[ancilla_key], center=tile.measurement_qubit)

    L = _logical_pauli_map(patch, meas_basis)
    builder.add_flow(start=L, end=L)
    return builder.finish_chunk()


def build_round_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    """Standard round: absorb + emit each stabilizer (two flows per tile)."""
    builder = sf.ChunkBuilder()
    append_round_body(builder, patch)

    for tile in patch.tiles:
        pm = _tile_pauli_map(tile)
        ancilla_key = tile.measurement_qubit
        builder.add_flow(start=pm, measurements=[ancilla_key], center=tile.measurement_qubit)
        builder.add_flow(end=pm, measurements=[ancilla_key], center=tile.measurement_qubit)

    L = _logical_pauli_map(patch, meas_basis)
    builder.add_flow(start=L, end=L)
    return builder.finish_chunk()


def build_final_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    """Destructive data measurement. Absorbs meas_basis stabs into data parities."""
    builder = sf.ChunkBuilder()
    data_coords = sorted(patch.data_set, key=coord_sort_key)
    meas_gate = "M" if meas_basis == "Z" else "MX"
    builder.append(meas_gate, data_coords)

    for tile in patch.tiles:
        pm = _tile_pauli_map(tile)
        if tile.basis == meas_basis:
            data_keys = [c for c in tile.data_qubits.values() if c is not None]
            builder.add_flow(start=pm, measurements=data_keys, center=tile.measurement_qubit)
        else:
            builder.add_discarded_flow_input(pm)

    strip = _logical_strip_coords(patch, meas_basis)
    builder.add_flow(start=_logical_pauli_map(patch, meas_basis), measurements=strip)
    return builder.finish_chunk()


def build_memory_chunks(
    x_distance: int, z_distance: int, rounds: int, meas_basis: str
) -> List[sf.Chunk]:
    if rounds < 1:
        raise ValueError(f"memory requires rounds >= 1, got {rounds}.")
    patch = rectangular_surface_code_patch(x_distance, z_distance)

    init = build_init_chunk(patch, meas_basis)
    round_first = build_round_first_chunk(patch, meas_basis)
    final = build_final_chunk(patch, meas_basis)

    if rounds == 1:
        return [init, round_first, final]
    round_normal = build_round_chunk(patch, meas_basis)
    return [init, round_first] + [round_normal] * (rounds - 1) + [final]


def memory(x_distance: int, z_distance: int, rounds: int, meas_basis: str) -> stim.Circuit:
    chunks = build_memory_chunks(x_distance, z_distance, rounds, meas_basis)
    compiler = sf.ChunkCompiler()
    for chunk in chunks:
        compiler.append(chunk)
    return compiler.finish_circuit()
