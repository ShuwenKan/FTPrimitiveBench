"""Chunks-backend lattice_surgery primitive.

Strategy: combine init + ancilla_init + transition into one chunk per phase
boundary; declare flows manually (without per-chunk verify) and rely on
DEM-parity tests to confirm correctness.

Per Phase 3 plan: handles pre_rounds=0 + merge_rounds>=1 + post_rounds=0 first;
extends incrementally.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import stim
import stimflow as sf

from .._make_circuit import (
    _build_merged_patch_components,
    _logical_bridge_measurement_positions,
)
from .._tile import Patch
from .._utils import coord_sort_key
from ._round import append_round_body

_OBS_NAME = "L"
_FLIP = {"X": "Z", "Z": "X"}


def _tile_pauli(tile) -> sf.PauliMap:
    return sf.PauliMap(
        {c: tile.basis for c in tile.data_qubits.values() if c is not None}
    )


def _classify(merged: Patch, indiv: Patch) -> Dict[complex, Tuple[str, FrozenSet[complex], FrozenSet[complex], str]]:
    indv_by_m = {t.measurement_qubit: t for t in indiv.tiles}
    cls = {}
    for mt in merged.tiles:
        m = mt.measurement_qubit
        D_m = frozenset(d for d in mt.data_qubits.values() if d is not None)
        it = indv_by_m.get(m)
        if it is None:
            cls[m] = ("C", frozenset(), D_m, mt.basis)
        else:
            D_i = frozenset(d for d in it.data_qubits.values() if d is not None)
            extra = D_m - D_i
            cls[m] = (("A" if not extra else "B"), D_i, extra, mt.basis)
    return cls


def _logical_strip(patch: Patch, basis: str) -> dict[complex, str]:
    data = sorted(patch.data_set, key=coord_sort_key)
    if not data:
        return {}
    xs = sorted({q.real for q in data})
    ys = sorted({q.imag for q in data})
    mid_x = xs[len(xs) // 2]
    mid_y = ys[len(ys) // 2]
    if basis == "Z":
        return {q: basis for q in data if q.real == mid_x}
    return {q: basis for q in data if q.imag == mid_y}


def _logical_pm(first: Patch, second: Patch, basis: str) -> sf.PauliMap:
    d1 = _logical_strip(first, basis)
    d2 = _logical_strip(second, basis)
    combined = {**d1, **d2}
    return sf.PauliMap(combined).with_obs_name(_OBS_NAME)


def build_pre_round_first_chunk(
    first: Patch, second: Patch, indiv: Patch, meas_basis: str
) -> sf.Chunk:
    """First pre-merge round on combined individual patch. Emits Z+X open flows."""
    b = sf.ChunkBuilder()
    append_round_body(b, indiv)
    for tile in indiv.tiles:
        pm = _tile_pauli(tile)
        m = tile.measurement_qubit
        if tile.basis == meas_basis:
            b.add_flow(start=pm, measurements=[m], center=m)
        b.add_flow(end=pm, measurements=[m], center=m)
    L = _logical_pm(first, second, meas_basis)
    b.add_flow(start=L, end=L)
    return b.finish_chunk()


def build_pre_round_chunk(
    first: Patch, second: Patch, indiv: Patch, meas_basis: str
) -> sf.Chunk:
    """Standard pre-merge round; absorb+emit all individual stabs."""
    b = sf.ChunkBuilder()
    append_round_body(b, indiv)
    for tile in indiv.tiles:
        pm = _tile_pauli(tile)
        m = tile.measurement_qubit
        b.add_flow(start=pm, measurements=[m], center=m)
        b.add_flow(end=pm, measurements=[m], center=m)
    L = _logical_pm(first, second, meas_basis)
    b.add_flow(start=L, end=L)
    return b.finish_chunk()


def build_init_chunk(first: Patch, second: Patch, indiv: Patch, meas_basis: str) -> sf.Chunk:
    b = sf.ChunkBuilder()
    init_gate = "R" if meas_basis == "Z" else "RX"
    b.append(init_gate, sorted(indiv.data_set, key=coord_sort_key))
    for t in indiv.tiles:
        pm = _tile_pauli(t)
        if t.basis == meas_basis:
            b.add_flow(end=pm, center=t.measurement_qubit)
        else:
            b.add_discarded_flow_output(pm)
    b.add_flow(end=_logical_pm(first, second, meas_basis))
    return b.finish_chunk()


def build_transition_chunk(
    first: Patch,
    second: Patch,
    indiv: Patch,
    merged: Patch,
    meas_basis: str,
    *,
    is_first_in_circuit: bool,
    emit_observable: bool,
) -> sf.Chunk:
    """One merged round with TYPE A/B/C transition flows, ancilla init, optional L readout."""
    ancilla_basis = _FLIP[meas_basis]
    cls = _classify(merged, indiv)
    ancilla_data = sorted(set(merged.data_set) - set(indiv.data_set), key=coord_sort_key)

    b = sf.ChunkBuilder()
    anc_init_gate = "RX" if meas_basis == "Z" else "R"
    if ancilla_data:
        b.append(anc_init_gate, ancilla_data)
        b.append("TICK")
    append_round_body(b, merged)

    for tile in merged.tiles:
        m = tile.measurement_qubit
        ttype, D_i, D_anc_extra, basis = cls[m]
        merged_pm = _tile_pauli(tile)
        if ttype == "A":
            # Merged == individual. For is_first_in_circuit: init emitted only
            # meas_basis tiles. For pre_rounds>0: pre_round_first emitted both
            # bases. So absorb when we have a prior open flow.
            if is_first_in_circuit:
                if basis == meas_basis:
                    b.add_flow(start=merged_pm, measurements=[m], center=m)
            else:
                # Both bases have open flows from pre rounds.
                b.add_flow(start=merged_pm, measurements=[m], center=m)
            b.add_flow(end=merged_pm, measurements=[m], center=m)
        elif ttype == "B":
            # Merged = individual * D_anc_extra.
            individual_pm = sf.PauliMap({c: basis for c in D_i})
            if is_first_in_circuit:
                if basis == meas_basis:
                    # Init has end=individual_Z. Random in transition.
                    b.add_discarded_flow_input(individual_pm)
                # basis == ancilla_basis: init discarded individual_X output. No prior.
            else:
                # Pre rounds emitted both bases.
                if basis == meas_basis:
                    # Merged Z is random; discard prev individual Z.
                    b.add_discarded_flow_input(individual_pm)
                else:
                    # basis == ancilla_basis: the merged_X stab measurement records
                    # X parity of (D_i ∪ D_anc_extra). RX init makes X on D_anc_extra
                    # deterministic +1, so the recorded value equals X parity of D_i =
                    # individual_X_stab. Declaring absorb directly closes the prev
                    # individual_X flow → 2-rec detector.
                    b.add_flow(
                        start=individual_pm, measurements=[m], center=m
                    )
            b.add_flow(end=merged_pm, measurements=[m], center=m)
        else:  # TYPE C
            if basis == ancilla_basis:
                # Deterministic single-rec detector at transition time
                # (X-stab is +1 from ancilla data init via RX).
                b.add_flow(measurements=[m], center=m)
            b.add_flow(end=merged_pm, measurements=[m], center=m)

    if emit_observable:
        merged_z = sorted((t.measurement_qubit for t in merged.tiles if t.basis == "Z"), key=coord_sort_key)
        merged_x = sorted((t.measurement_qubit for t in merged.tiles if t.basis == "X"), key=coord_sort_key)
        sel = _logical_bridge_measurement_positions(
            first_patch=first, second_patch=second,
            merged_meas_x_coords=merged_x, merged_meas_z_coords=merged_z,
            meas_basis=meas_basis,
        )
        coord_list = merged_x if meas_basis == "X" else merged_z
        bridge_anc_keys = [coord_list[i] for i in sel]
        L = _logical_pm(first, second, meas_basis)
        b.add_flow(start=L, measurements=bridge_anc_keys)
    else:
        # Passthrough L
        L = _logical_pm(first, second, meas_basis)
        b.add_flow(start=L, end=L)

    return b.finish_chunk()


def build_merge_round_chunk(merged: Patch, first: Patch, second: Patch, meas_basis: str) -> sf.Chunk:
    """Standard merge round (not first, not last). Passthrough all merged stabs and L."""
    b = sf.ChunkBuilder()
    append_round_body(b, merged)
    for tile in merged.tiles:
        pm = _tile_pauli(tile)
        m = tile.measurement_qubit
        b.add_flow(start=pm, measurements=[m], center=m)
        b.add_flow(end=pm, measurements=[m], center=m)
    L = _logical_pm(first, second, meas_basis)
    b.add_flow(start=L, end=L)
    return b.finish_chunk()


def build_merge_last_chunk(merged: Patch, first: Patch, second: Patch, meas_basis: str) -> sf.Chunk:
    """Last merge round: passthrough all merged stabs, emit observable readout."""
    b = sf.ChunkBuilder()
    append_round_body(b, merged)
    for tile in merged.tiles:
        pm = _tile_pauli(tile)
        m = tile.measurement_qubit
        b.add_flow(start=pm, measurements=[m], center=m)
        b.add_flow(end=pm, measurements=[m], center=m)
    merged_z = sorted((t.measurement_qubit for t in merged.tiles if t.basis == "Z"), key=coord_sort_key)
    merged_x = sorted((t.measurement_qubit for t in merged.tiles if t.basis == "X"), key=coord_sort_key)
    sel = _logical_bridge_measurement_positions(
        first_patch=first, second_patch=second,
        merged_meas_x_coords=merged_x, merged_meas_z_coords=merged_z,
        meas_basis=meas_basis,
    )
    coord_list = merged_x if meas_basis == "X" else merged_z
    bridge_anc_keys = [coord_list[i] for i in sel]
    L = _logical_pm(first, second, meas_basis)
    b.add_flow(start=L, measurements=bridge_anc_keys)
    return b.finish_chunk()


def build_ancilla_destruct_chunk(
    merged: Patch, indiv: Patch, meas_basis: str, *, has_post: bool
) -> sf.Chunk:
    """Destructive ancilla data measurement.

    - TYPE A meas_basis: always pass through (final absorbs).
    - TYPE A ancilla_basis: pass through if has_post (post_first absorbs); else discard.
    - TYPE B meas_basis: discard (legacy doesn't emit detector at this boundary).
    - TYPE B ancilla_basis: if has_post, transform merged_pm → individual_pm via
        ancilla destruct recs (X parity on D_anc_extra is recorded by MX); else discard.
    - TYPE C ancilla_basis: absorb merged_pm via ancilla destruct recs.
    - TYPE C meas_basis: discard.
    """
    ancilla_basis = _FLIP[meas_basis]
    cls = _classify(merged, indiv)
    ancilla_data_set = set(merged.data_set) - set(indiv.data_set)
    ancilla_data = sorted(ancilla_data_set, key=coord_sort_key)
    b = sf.ChunkBuilder()
    anc_destruct_gate = "MX" if meas_basis == "Z" else "M"
    if ancilla_data:
        b.append(anc_destruct_gate, ancilla_data)

    for tile in merged.tiles:
        m = tile.measurement_qubit
        ttype, D_i, D_anc_extra, basis = cls[m]
        merged_pm = _tile_pauli(tile)

        if ttype == "A":
            if basis == meas_basis or has_post:
                # Pass through unchanged.
                b.add_flow(start=merged_pm, end=merged_pm, center=m)
            else:
                b.add_discarded_flow_input(merged_pm)
        elif ttype == "B":
            if basis == ancilla_basis and has_post:
                # Transform merged_X_pm → individual_X_pm by recording bridge X parity.
                individual_pm = sf.PauliMap({c: basis for c in D_i})
                keys = [d for d in D_anc_extra if d in ancilla_data_set]
                if keys:
                    b.add_flow(start=merged_pm, end=individual_pm, measurements=keys, center=m)
                else:
                    b.add_discarded_flow_input(merged_pm)
            else:
                b.add_discarded_flow_input(merged_pm)
        else:  # TYPE C
            if basis == ancilla_basis:
                keys = [d for d in tile.data_qubits.values() if d is not None and d in ancilla_data_set]
                if keys:
                    b.add_flow(start=merged_pm, measurements=keys, center=m)
                else:
                    b.add_discarded_flow_input(merged_pm)
            else:
                b.add_discarded_flow_input(merged_pm)

    return b.finish_chunk()


def build_post_round_chunk(indiv: Patch, meas_basis: str) -> sf.Chunk:
    """Post-merge individual round; absorb+emit both bases."""
    b = sf.ChunkBuilder()
    append_round_body(b, indiv)
    for tile in indiv.tiles:
        pm = _tile_pauli(tile)
        m = tile.measurement_qubit
        b.add_flow(start=pm, measurements=[m], center=m)
        b.add_flow(end=pm, measurements=[m], center=m)
    return b.finish_chunk()


def build_final_chunk(indiv: Patch, meas_basis: str, *, has_post: bool) -> sf.Chunk:
    """Destructive data measurement. Absorbs meas_basis stabs; discards opp basis if has_post."""
    ancilla_basis = _FLIP[meas_basis]
    b = sf.ChunkBuilder()
    data = sorted(indiv.data_set, key=coord_sort_key)
    meas_gate = "M" if meas_basis == "Z" else "MX"
    b.append(meas_gate, data)
    for t in indiv.tiles:
        pm = _tile_pauli(t)
        if t.basis == meas_basis:
            keys = [c for c in t.data_qubits.values() if c is not None]
            b.add_flow(start=pm, measurements=keys, center=t.measurement_qubit)
        elif has_post:
            # Open ancilla_basis flows from post rounds need to be closed.
            b.add_discarded_flow_input(pm)
    return b.finish_chunk()


def build_lattice_surgery_chunks(
    x_distance: int,
    z_distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    post_rounds: int,
    meas_basis: str,
) -> List[sf.Chunk]:
    horizontal = meas_basis == "X"
    stack_dim = x_distance if horizontal else z_distance
    if (stack_dim + bridge_length) % 2 != 0:
        raise ValueError(
            f"lattice_surgery requires ({'x' if horizontal else 'z'}_distance "
            f"+ bridge_length) to be even."
        )

    boundary = {"top": "X", "bot": "X", "left": "Z", "right": "Z"}
    first, second, indiv, merged = _build_merged_patch_components(
        x_distance, z_distance, bridge_length,
        horizontal=horizontal,
        individual_boundaries=boundary,
        merged_boundaries=boundary,
    )
    indiv._invalidate_cached_sets()

    chunks: List[sf.Chunk] = [build_init_chunk(first, second, indiv, meas_basis)]
    if pre_rounds > 0:
        chunks.append(build_pre_round_first_chunk(first, second, indiv, meas_basis))
        chunks += [build_pre_round_chunk(first, second, indiv, meas_basis)] * (pre_rounds - 1)
    if merge_rounds == 1:
        chunks.append(build_transition_chunk(
            first, second, indiv, merged, meas_basis,
            is_first_in_circuit=(pre_rounds == 0), emit_observable=True,
        ))
    else:
        chunks.append(build_transition_chunk(
            first, second, indiv, merged, meas_basis,
            is_first_in_circuit=(pre_rounds == 0), emit_observable=False,
        ))
        chunks += [build_merge_round_chunk(merged, first, second, meas_basis)] * max(0, merge_rounds - 2)
        chunks.append(build_merge_last_chunk(merged, first, second, meas_basis))
    has_post = post_rounds > 0
    chunks.append(build_ancilla_destruct_chunk(merged, indiv, meas_basis, has_post=has_post))
    if has_post:
        chunks += [build_post_round_chunk(indiv, meas_basis)] * post_rounds
    chunks.append(build_final_chunk(indiv, meas_basis, has_post=has_post))
    return chunks


def lattice_surgery(
    x_distance: int,
    z_distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    post_rounds: int,
    meas_basis: str,
) -> stim.Circuit:
    chunks = build_lattice_surgery_chunks(
        x_distance, z_distance, bridge_length, pre_rounds, merge_rounds, post_rounds, meas_basis
    )
    compiler = sf.ChunkCompiler()
    for chunk in chunks:
        compiler.append(chunk)
    return compiler.finish_circuit()
