"""Chunks-backend transversal_h primitive.

Decomposition::

    init  +  round_first  +  pre_round * pre_rounds  +  h_layer
         +  post_round * post_rounds  +  final

The H-layer chunk applies H transversally to every data qubit. Each
stabilizer's Pauli letter on data flips (Z ↔ X); the same coordinates
remain involved. Post-H rounds use the swap_xz_roles=True schedule
(matches legacy bit-for-bit) and absorb/emit the FLIPPED-basis stabs.

The legacy primitive's "deterministic" first round is reproduced via
the same init + round_first pattern as memory. The first post-H round
is a standard chunks round on the flipped basis — the H chunk's
declared Z↔X flow transitions make the per-tile detector chain
continue across the H layer.
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
_FLIP = {"X": "Z", "Z": "X"}


def _logical_strip_coords(patch: Patch, basis_for_shape: str) -> List[complex]:
    """Midline data strip oriented for the given basis.

    The strip *shape* is invariant under H — only the Pauli letters on it flip.
    Z-basis strip is a vertical column (constant x); X-basis is horizontal.
    """
    data = sorted(patch.data_set, key=coord_sort_key)
    if not data:
        return []
    xs = [round(c2xy(q)[0]) for q in data]
    ys = [round(c2xy(q)[1]) for q in data]
    mid_x = (min(xs) + max(xs)) // 2
    mid_y = (min(ys) + max(ys)) // 2
    if basis_for_shape == "Z":
        return [q for q in data if round(c2xy(q)[0]) == mid_x]
    return [q for q in data if round(c2xy(q)[1]) == mid_y]


def _logical_pauli_map(patch: Patch, shape_basis: str, letter_basis: str) -> sf.PauliMap:
    """PauliMap on the shape_basis strip, with Pauli letter letter_basis."""
    strip = _logical_strip_coords(patch, shape_basis)
    return sf.PauliMap({q: letter_basis for q in strip}).with_obs_name(_OBS_NAME)


def _tile_pauli_map(tile, basis_letter: str) -> sf.PauliMap:
    """Pauli on the tile's data qubits using the given basis letter."""
    return sf.PauliMap(
        {coord: basis_letter for coord in tile.data_qubits.values() if coord is not None}
    )


def build_init_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    builder = sf.ChunkBuilder()
    data_coords = sorted(patch.data_set, key=coord_sort_key)
    init_gate = "R" if meas_basis == "Z" else "RX"
    builder.append(init_gate, data_coords)
    for tile in patch.tiles:
        pm = _tile_pauli_map(tile, tile.basis)
        if tile.basis == meas_basis:
            builder.add_flow(end=pm, center=tile.measurement_qubit)
        else:
            builder.add_discarded_flow_output(pm)
    builder.add_flow(end=_logical_pauli_map(patch, meas_basis, meas_basis))
    return builder.finish_chunk()


def build_round_first_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    builder = sf.ChunkBuilder()
    append_round_body(builder, patch)
    for tile in patch.tiles:
        pm = _tile_pauli_map(tile, tile.basis)
        ancilla_key = tile.measurement_qubit
        if tile.basis == meas_basis:
            builder.add_flow(start=pm, measurements=[ancilla_key], center=tile.measurement_qubit)
        builder.add_flow(end=pm, measurements=[ancilla_key], center=tile.measurement_qubit)
    L = _logical_pauli_map(patch, meas_basis, meas_basis)
    builder.add_flow(start=L, end=L)
    return builder.finish_chunk()


def build_pre_round_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    builder = sf.ChunkBuilder()
    append_round_body(builder, patch)
    for tile in patch.tiles:
        pm = _tile_pauli_map(tile, tile.basis)
        ancilla_key = tile.measurement_qubit
        builder.add_flow(start=pm, measurements=[ancilla_key], center=tile.measurement_qubit)
        builder.add_flow(end=pm, measurements=[ancilla_key], center=tile.measurement_qubit)
    L = _logical_pauli_map(patch, meas_basis, meas_basis)
    builder.add_flow(start=L, end=L)
    return builder.finish_chunk()


def build_h_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    """Transversal H on all data qubits. Flips Z ↔ X on every stabilizer and on L."""
    builder = sf.ChunkBuilder()
    data_coords = sorted(patch.data_set, key=coord_sort_key)
    builder.append("H", data_coords)
    for tile in patch.tiles:
        b = tile.basis
        b_flipped = _FLIP[b]
        builder.add_flow(
            start=_tile_pauli_map(tile, b),
            end=_tile_pauli_map(tile, b_flipped),
            center=tile.measurement_qubit,
        )
    L_in = _logical_pauli_map(patch, meas_basis, meas_basis)
    L_out = _logical_pauli_map(patch, meas_basis, _FLIP[meas_basis])
    builder.add_flow(start=L_in, end=L_out)
    return builder.finish_chunk()


def build_post_round_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    """Post-H round: swap_xz_roles=True schedule, measure_z_first=True.

    Tiles still have their original basis labelling, but the Pauli on data is
    swapped: a "Z-tile" now measures an X-stab on data (since H flipped data).
    The flow's PauliMap uses the SWAPPED basis letters.
    """
    builder = sf.ChunkBuilder()
    append_round_body(builder, patch, swap_xz_roles=True, measure_z_first=True)
    for tile in patch.tiles:
        flipped_pm = _tile_pauli_map(tile, _FLIP[tile.basis])
        ancilla_key = tile.measurement_qubit
        builder.add_flow(
            start=flipped_pm, measurements=[ancilla_key], center=tile.measurement_qubit
        )
        builder.add_flow(
            end=flipped_pm, measurements=[ancilla_key], center=tile.measurement_qubit
        )
    L = _logical_pauli_map(patch, meas_basis, _FLIP[meas_basis])
    builder.add_flow(start=L, end=L)
    return builder.finish_chunk()


def build_final_chunk(patch: Patch, meas_basis: str) -> sf.Chunk:
    """Final destructive data measurement in the FLIPPED basis.

    After H, the data basis is swap(meas_basis). The L observable lives on the
    same strip but with flipped letters. Tiles of swap(meas_basis)-letter
    (originally meas_basis tiles) are deterministic in this final basis and
    absorbed into data parities.
    """
    builder = sf.ChunkBuilder()
    final_basis = _FLIP[meas_basis]
    data_coords = sorted(patch.data_set, key=coord_sort_key)
    meas_gate = "M" if final_basis == "Z" else "MX"
    builder.append(meas_gate, data_coords)
    for tile in patch.tiles:
        flipped_pm = _tile_pauli_map(tile, _FLIP[tile.basis])
        if tile.basis == meas_basis:
            # After H, this tile's stab is in `final_basis` letters → measurable.
            data_keys = [c for c in tile.data_qubits.values() if c is not None]
            builder.add_flow(
                start=flipped_pm, measurements=data_keys, center=tile.measurement_qubit
            )
        else:
            builder.add_discarded_flow_input(flipped_pm)
    strip = _logical_strip_coords(patch, meas_basis)
    L = _logical_pauli_map(patch, meas_basis, final_basis)
    builder.add_flow(start=L, measurements=strip)
    return builder.finish_chunk()


def build_transversal_h_chunks(
    x_distance: int, z_distance: int, pre_rounds: int, post_rounds: int, meas_basis: str
) -> List[sf.Chunk]:
    patch = rectangular_surface_code_patch(x_distance, z_distance)
    init = build_init_chunk(patch, meas_basis)
    round_first = build_round_first_chunk(patch, meas_basis)
    pre = build_pre_round_chunk(patch, meas_basis)
    h = build_h_chunk(patch, meas_basis)
    post = build_post_round_chunk(patch, meas_basis)
    final = build_final_chunk(patch, meas_basis)

    chunks: List[sf.Chunk] = [init, round_first]
    chunks += [pre] * max(0, pre_rounds)
    chunks.append(h)
    chunks += [post] * max(0, post_rounds)
    chunks.append(final)
    return chunks


def transversal_h(
    x_distance: int,
    z_distance: int,
    pre_rounds: int,
    post_rounds: int,
    meas_basis: str,
) -> stim.Circuit:
    chunks = build_transversal_h_chunks(x_distance, z_distance, pre_rounds, post_rounds, meas_basis)
    compiler = sf.ChunkCompiler()
    for chunk in chunks:
        compiler.append(chunk)
    return compiler.finish_circuit()
